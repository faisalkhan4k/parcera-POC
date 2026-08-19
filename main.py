import asyncio
import os
import json
import base64
import time
import sounddevice as sd
import websockets
from dotenv import load_dotenv
from groq import AsyncGroq
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions

from core.state import SessionManager, redis_client
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

# --- Audio Configuration ---
INPUT_RATE = 16000
OUTPUT_RATE = 16000 
CHANNELS = 1

speaker_stream = sd.RawOutputStream(
    samplerate=OUTPUT_RATE,
    channels=CHANNELS,
    dtype='int16'
)
speaker_stream.start()

# --- PHASE 4 Telemetry Tracker ---
telemetry = {
    "total_turns": 0,
    "total_latency_ms": 0,
    "silent_failures_caught": 0,
    "repetitive_asks_blocked": 0
}

# --- Global State & Queues ---
is_agent_speaking = False
active_task = None
audio_queue = asyncio.Queue()
main_loop = None  
speech_start_time = 0  
session = None
current_turn_id = 0  

# Persistent Base Identity
BASE_SYSTEM_PROMPT = (
    "You are the phone ordering AI assistant for Atomic Wings restaurant. "
    "Speak naturally in 1 short, direct sentence suitable for voice. "
    "Never offer general assistant tasks like writing or coding. "
    "Never guess, invent prices, or invent menu items. If a user asks for an item or detail "
    "not explicitly provided in your Current Task, state that you do not have that information. "
    "Do not use markdown tables or bullet points."
)

conversation_history = [
    {"role": "system", "content": BASE_SYSTEM_PROMPT}
]

def audio_callback(indata, frames, time_info, status):
    if main_loop is not None:
        main_loop.call_soon_threadsafe(audio_queue.put_nowait, bytes(indata))

async def stream_to_elevenlabs(text_iterator, turn_id):
    global is_agent_speaking
    
    voice_id = "CwhRBWXzGAHq8TQ4Fs17" 
    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id=eleven_turbo_v2_5&output_format=pcm_16000&optimize_streaming_latency=3"
    headers = {"xi-api-key": ELEVENLABS_KEY}
    
    try:
        async with websockets.connect(uri, additional_headers=headers) as websocket:
            await websocket.send(json.dumps({
                "text": " ", 
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}
            }))

            async def send_text():
                try:
                    async for text_chunk in text_iterator:
                        if not is_agent_speaking or turn_id != current_turn_id: 
                            break  
                        await websocket.send(json.dumps({
                            "text": text_chunk, 
                            "try_trigger_generation": True
                        }))
                finally:
                    try:
                        await websocket.send(json.dumps({"text": ""}))
                    except websockets.exceptions.ConnectionClosed:
                        pass

            async def receive_audio():
                while is_agent_speaking and turn_id == current_turn_id:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                        data = json.loads(message)
                        
                        if "error" in data:
                            print(f"\n❌ [ElevenLabs API Error]: {data['error']}")
                            break

                        if data.get("audio"):
                            raw_pcm = base64.b64decode(data["audio"])
                            if len(raw_pcm) % 2 != 0:
                                raw_pcm += b'\x00'
                            await asyncio.to_thread(speaker_stream.write, raw_pcm)
                            
                        if data.get("isFinal"):
                            break
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        break
                    except Exception as e:
                        print(f"\n❌ [Audio Playback Error]: {e}")
                        break

            await asyncio.gather(send_text(), receive_audio())
            
    except Exception as e:
        print(f"\n❌ [ElevenLabs Connection Error]: {e}")


async def process_user_turn(user_text, turn_id):
    """The 4-Node Linear Pipeline Orchestrator"""
    global is_agent_speaking, speech_start_time, conversation_history, session, current_turn_id
    
    try:
        is_agent_speaking = True
        speech_start_time = time.time() 
        
        print(f"\n[You]: {user_text}")
        print("[Agent]: ", end="", flush=True)

        # --- NODE B: Fetch State & Cart ---
        current_state = await session.get_state() or "GREETING"
        cart_ids = await session.get_cart()

        # --- NODE A: Semantic Routing ---
        matched_dict, score = await match_by_meaning(user_text)

        # --- NODE C: Deterministic Rule Engine ---
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
                # --- PHASE 2 HARD RULE: Prevent Empty Checkout ---
                if not cart_ids:
                    next_state = current_state  
                    turn_instruction = "The customer tried to checkout, but their cart is completely empty. Tell them they need to add items to their order before checking out."
                else:
                    next_state = "CHECKOUT"
                    cart_total = calculate_total(cart_ids)
                    turn_instruction = f"The customer is done ordering. Current cart is {cart_summary}. Total is ${cart_total:.2f}. Ask for final confirmation to place the order."  

        elif matched_dict["type"] == "ITEM":
            # Guard: Check if the user is asking a question vs. placing an order
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

        # Observability Dashboard
        print(f"\n" + "="*55)
        print(f"🎙️  [STT Transcript] : '{user_text}'")
        print(f"🧠  [Node A Match]   : [{matched_dict['type']}] {matched_dict['id']} ({score:.2f})")
        print(f"🚦  [Node B Cart]    : {cart_ids}")
        print(f"📜  [Node C Target]  : {turn_instruction[:50]}...")
        print(f"⏱️  [Node Latency]   : {round((time.time() - speech_start_time)*1000, 2)} ms")
        print("="*55)

        # --- NODE D: Narrow Generation ---
        active_system_prompt = f"{BASE_SYSTEM_PROMPT}\nCurrent Task: {turn_instruction}"
        conversation_history[0] = {"role": "system", "content": active_system_prompt}
        conversation_history.append({"role": "user", "content": user_text})
        
        # --- PHASE 3: Silent-Failure Guard (2.5s Deadline) ---
        stream = await asyncio.wait_for(
            llm_client.chat.completions.create(
                model="openai/gpt-oss-20b", 
                messages=conversation_history,
                stream=True,
                max_tokens=100,
                reasoning_effort="low"
            ),
            timeout=2.5 
        )
        
        agent_response = ""
        async def token_generator():
            nonlocal agent_response
            async for chunk in stream:
                if not is_agent_speaking or turn_id != current_turn_id: 
                    break
                token = chunk.choices[0].delta.content
                if token: 
                    agent_response += token
                    print(token, end="", flush=True)
                    yield token
            print("\n")

        await stream_to_elevenlabs(token_generator(), turn_id)
        
        if agent_response and turn_id == current_turn_id:
            conversation_history.append({"role": "assistant", "content": agent_response})
            
    except asyncio.TimeoutError:
        print("\n⚠️ [Phase 3 Guard] LLM Timeout. Catching dead air...")
        telemetry["silent_failures_caught"] += 1
        async def fallback_generator():
            yield "Bear with me just one moment..."
        await stream_to_elevenlabs(fallback_generator(), turn_id)

    except asyncio.CancelledError:
        pass  # Expected on barge-in

    except Exception as e:
        print(f"\n❌ [Pipeline Error]: {e}")

    finally:
        if turn_id == current_turn_id:
            is_agent_speaking = False
            
        # Log metrics for the Phase 4 demo
        telemetry["total_turns"] += 1
        turn_latency = (time.time() - speech_start_time) * 1000
        telemetry["total_latency_ms"] += turn_latency
        
        print("\n" + "-"*30)
        print(f"📊 [TELEMETRY] Turns: {telemetry['total_turns']} | Avg Latency: {round(telemetry['total_latency_ms'] / telemetry['total_turns'], 2)} ms")
        print("-"*30)
            

def on_message(self, result, **kwargs):
    global is_agent_speaking, active_task, main_loop, current_turn_id

    transcript = result.channel.alternatives[0].transcript
    elapsed = time.time() - speech_start_time

    # Barge-In Trigger
    if not result.is_final and transcript and is_agent_speaking:
        if elapsed > 1.0 and len(transcript.strip()) > 3:
            is_agent_speaking = False
            current_turn_id += 1  
            if active_task and not active_task.done():
                active_task.cancel()
            print("\n[Interrupted!]")

    # Turn Execution
    if result.is_final and transcript and transcript.strip():
        current_turn_id += 1
        turn_to_run = current_turn_id
        
        if active_task and not active_task.done():
            active_task.cancel()
            
        if main_loop is not None:
            active_task = asyncio.run_coroutine_threadsafe(
                process_user_turn(transcript, turn_to_run), main_loop
            )


async def main():
    global main_loop, session
    main_loop = asyncio.get_running_loop()
    
    init_db()
    session = SessionManager(session_id="POC_DEMO_001")
    await session.initialize_call()

    dg_connection = dg_client.listen.websocket.v("1")
    dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

    options = LiveOptions(
        model="nova-2",
        language="en-US",
        smart_format=True,
        endpointing=600,
        interim_results=True,
        encoding="linear16",      
        sample_rate=INPUT_RATE,   
        channels=CHANNELS,        
    )

    if not dg_connection.start(options):
        print("Failed to connect to Deepgram.")
        return

    print("\n🟢 Pipeline Active! Atomic Wings AI Initialized. Speak into your microphone...")
    print("⚠️  Ensure you are wearing headphones to prevent acoustic echo interruptions.")

    mic_stream = sd.RawInputStream(
        samplerate=INPUT_RATE,
        blocksize=1024,
        channels=CHANNELS,
        dtype='int16',
        callback=audio_callback
    )

    with mic_stream:
        try:
            while True:
                data = await audio_queue.get()
                dg_connection.send(data)
                audio_queue.task_done()
        except KeyboardInterrupt:
            print("\nShutting down pipeline...")
        finally:
            dg_connection.finish()
            speaker_stream.stop()
            speaker_stream.close()

if __name__ == "__main__":
    asyncio.run(main())