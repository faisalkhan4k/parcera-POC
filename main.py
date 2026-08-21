import asyncio
import os
import json
import base64
import time
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from groq import AsyncGroq
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions

from core.state import SessionManager
from nodes.router import match_by_meaning, get_item_name
from nodes.database import init_db, calculate_total, get_item_price

load_dotenv()

# --- API Keys ---
DG_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
ELEVENLABS_KEY = os.getenv("ELEVENLABS_API_KEY")

# --- Clients ---
dg_client = DeepgramClient(DG_KEY)
llm_client = AsyncGroq(api_key=GROQ_KEY)

INPUT_RATE = 16000

app = FastAPI()
templates = Jinja2Templates(directory="templates")

BASE_SYSTEM_PROMPT = (
    "You are the phone ordering AI assistant for Atomic Wings restaurant. "
    "Speak naturally in 1 short, direct sentence suitable for voice. "
    "Never offer general assistant tasks like writing or coding. "
    "Never guess, invent prices, or invent menu items. If a user asks for an item or detail "
    "not explicitly provided in your Current Task, state that you do not have that information. "
    "Do not use markdown tables or bullet points."
)


@app.on_event("startup")
async def startup_event():
    init_db()


@app.get("/", response_class=HTMLResponse)
async def get_webpage(request: Request):
    # Newer Starlette requires `request` as the first positional argument.
    return templates.TemplateResponse(request, "index.html")


@app.get("/favicon.ico")
async def favicon():
    # Prevents the browser's automatic favicon request from showing as a 404
    # in the console -- harmless on its own, but worth silencing so real
    # errors are easy to spot during remote testing.
    return Response(status_code=204)


class ConnectionState:
    """Holds all per-connection state for one caller's continuous conversation.
    A fresh instance is created per browser tab, so multiple simultaneous
    testers don't share or clobber each other's state."""

    def __init__(self, websocket: WebSocket, loop: asyncio.AbstractEventLoop):
        self.websocket = websocket
        self.loop = loop
        self.is_agent_speaking = False
        self.active_task = None
        self.speech_start_time = 0
        self.dg_connection = None
        self.session = SessionManager(session_id=f"WEB_DEMO_{int(time.time())}")
        self.conversation_history = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]

    def start_deepgram(self):
        self.dg_connection = dg_client.listen.websocket.v("1")

        def on_message(_, result, **kwargs):
            transcript = result.channel.alternatives[0].transcript
            elapsed = time.time() - self.speech_start_time

            if not result.is_final and transcript and self.is_agent_speaking:
                if elapsed > 1.2 and len(transcript.strip()) > 5:
                    self.is_agent_speaking = False
                    asyncio.run_coroutine_threadsafe(self._notify_interrupted(), self.loop)

            if result.is_final and transcript:
                if self.active_task and not self.active_task.done():
                    self.active_task.cancel()
                self.active_task = asyncio.run_coroutine_threadsafe(
                    process_user_turn(transcript, self), self.loop
                )

        self.dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

        options = LiveOptions(
            model="nova-2",
            language="en-US",
            smart_format=True,
            endpointing=500,
            interim_results=True,
            encoding="linear16",
            sample_rate=INPUT_RATE,
            channels=1,
        )
        self.dg_connection.start(options)

    async def _notify_interrupted(self):
        try:
            await self.websocket.send_text(json.dumps({"type": "interrupted"}))
        except Exception:
            pass

    def send_audio(self, data: bytes):
        if self.dg_connection:
            self.dg_connection.send(data)

    def close(self):
        if self.dg_connection:
            self.dg_connection.finish()


async def stream_to_elevenlabs_websocket(text_iterator, conn: ConnectionState):
    voice_id = "CwhRBWXzGAHq8TQ4Fs17"
    uri = (
        f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
        f"?model_id=eleven_turbo_v2_5&output_format=pcm_16000&optimize_streaming_latency=3"
    )
    headers = {"xi-api-key": ELEVENLABS_KEY}

    try:
        async with websockets.connect(uri, additional_headers=headers) as el_ws:
            await el_ws.send(json.dumps({
                "text": " ",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
            }))

            async def send_text():
                try:
                    async for text_chunk in text_iterator:
                        if not conn.is_agent_speaking:
                            break
                        await el_ws.send(json.dumps({
                            "text": text_chunk,
                            "try_trigger_generation": True,
                        }))
                finally:
                    try:
                        await el_ws.send(json.dumps({"text": ""}))
                    except websockets.exceptions.ConnectionClosed:
                        pass

            async def receive_audio():
                while conn.is_agent_speaking:
                    try:
                        message = await asyncio.wait_for(el_ws.recv(), timeout=0.1)
                        data = json.loads(message)

                        if "error" in data:
                            print(f"\n❌ [ElevenLabs API Error]: {data['error']}")
                            break

                        if data.get("audio"):
                            raw_pcm = base64.b64decode(data["audio"])
                            if conn.is_agent_speaking:
                                await conn.websocket.send_bytes(raw_pcm)

                        if data.get("isFinal"):
                            break
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        break
                    except Exception as e:
                        print(f"\n❌ [Audio Streaming Error]: {e}")
                        break

            await asyncio.gather(send_text(), receive_audio())

    except Exception as e:
        print(f"\n❌ [ElevenLabs Connection Error]: {e}")


async def process_user_turn(user_text: str, conn: ConnectionState):
    try:
        conn.is_agent_speaking = True
        conn.speech_start_time = time.time()

        print(f"\n[You]: {user_text}")
        print("[Agent]: ", end="", flush=True)

        session = conn.session
        conversation_history = conn.conversation_history

        current_state = await session.get_state() or "GREETING"
        cart_ids = await session.get_cart()

        matched_dict, score = await match_by_meaning(user_text)

        next_state = current_state
        readable_cart = [get_item_name(item_id) for item_id in cart_ids]
        cart_summary = ", ".join(readable_cart) if readable_cart else "empty"

        if matched_dict["type"] == "INTENT":
            if matched_dict["id"] == "INT_GREETING":
                next_state = "TAKING_ORDER"
                turn_instruction = "Greet the customer warmly at Atomic Wings and ask what food they would like to order."

            elif matched_dict["id"] == "INT_INQUIRE_MENU":
                turn_instruction = (
                    "List our main items: Chicken Sandwich Combo, Tenders Combo, "
                    "Boneless or Traditional Wings Combos, Family Combo, Flat Bread Pizza, "
                    "and Waffle Fries. Ask what they would like to order."
                )

            elif matched_dict["id"] == "INT_CHECK_CART":
                turn_instruction = f"State the current cart items ({cart_summary}) clearly in one sentence and ask if they are ready to checkout or add more."

            elif matched_dict["id"] == "INT_CHECKOUT":
                if not cart_ids:
                    next_state = current_state
                    turn_instruction = "The customer tried to checkout, but their cart is completely empty. Tell them they need to add items to their order before checking out."
                else:
                    next_state = "CHECKOUT"
                    cart_total = calculate_total(cart_ids)
                    turn_instruction = f"The customer is done ordering. Current cart is {cart_summary}. Total is ${cart_total:.2f}. Ask for final confirmation to place the order."
            else:
                turn_instruction = "Acknowledge what you heard. Do NOT list any menu items. Ask them to be more specific about exactly what they want to order."

        elif matched_dict["type"] == "ITEM":
            is_inquiry = any(word in user_text.lower() for word in ["what", "how", "tell", "show", "explain", "describe"])
            item_desc = matched_dict.get("text", matched_dict["name"])
            item_price = get_item_price(matched_dict["id"])

            if is_inquiry:
                turn_instruction = f"The customer is asking about the {matched_dict['name']}. Explain it exactly using this description: '{item_desc}'. Do not add it to the cart yet. Ask if they want to order it."
            else:
                next_state = "TAKING_ORDER"
                await session.add_to_cart(matched_dict["id"])
                cart_ids = await session.get_cart()
                readable_cart = [get_item_name(item_id) for item_id in cart_ids]
                turn_instruction = f"Confirm adding {matched_dict['name']} (${item_price:.2f}) to their order. Current items: {', '.join(readable_cart)}. Ask what else they'd like."

        elif matched_dict["type"] == "MODIFIER":
            await session.add_to_cart(matched_dict["id"])
            mod_price = get_item_price(matched_dict["id"])
            price_text = f"(${mod_price:.2f})" if mod_price > 0 else "(Free)"
            turn_instruction = f"Confirm the modification {matched_dict['name']} {price_text}. Ask what else they want to add."

        else:
            turn_instruction = "Acknowledge what you heard. Do NOT list any menu items. Ask them to be more specific about exactly what they want to order."

        await session.update_state(next_state)

        active_system_prompt = f"{BASE_SYSTEM_PROMPT}\nCurrent Task: {turn_instruction}"
        conversation_history[0] = {"role": "system", "content": active_system_prompt}
        conversation_history.append({"role": "user", "content": user_text})

        await conn.websocket.send_text(json.dumps({"type": "user_text", "text": user_text}))

        stream = await asyncio.wait_for(
            llm_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=conversation_history,
                stream=True,
                max_tokens=100,
                reasoning_effort="low",
            ),
            timeout=5.0,
        )

        agent_response = ""

        async def token_generator():
            nonlocal agent_response
            async for chunk in stream:
                if not conn.is_agent_speaking:
                    break
                token = chunk.choices[0].delta.content
                if token:
                    agent_response += token
                    print(token, end="", flush=True)
                    yield token
            print()

        await stream_to_elevenlabs_websocket(token_generator(), conn)

        if agent_response:
            conversation_history.append({"role": "assistant", "content": agent_response})
            await conn.websocket.send_text(json.dumps({"type": "ai_text", "text": agent_response}))

    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"\n❌ [Turn Error]: {e}")
    finally:
        conn.is_agent_speaking = False


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_running_loop()

    conn = ConnectionState(websocket, loop)
    await conn.session.initialize_call()
    conn.start_deepgram()
    print("\n[New tester connected]")

    try:
        while True:
            audio_bytes = await websocket.receive_bytes()
            conn.send_audio(audio_bytes)
    except WebSocketDisconnect:
        print("\n[Tester disconnected]")
    except Exception as e:
        print(f"\n❌ [WebSocket Error]: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)