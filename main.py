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

# --- Global State & Queues ---
is_agent_speaking = False
active_task = None
audio_queue = asyncio.Queue()
main_loop = None  
speech_start_time = 0  

conversation_history = [
    {"role": "system", "content": "You are a fast, concise drive-thru ordering assistant. Respond in 1 brief sentence."}
]

def audio_callback(indata, frames, time_info, status):
    if status:
        pass
    if main_loop is not None:
        main_loop.call_soon_threadsafe(audio_queue.put_nowait, bytes(indata))

async def stream_to_elevenlabs(text_iterator):
    global is_agent_speaking
    
    # YOUR specific custom voice ID
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
                        if not is_agent_speaking: 
                            break  
                        await websocket.send(json.dumps({
                            "text": text_chunk, 
                            "try_trigger_generation": True
                        }))
                finally:
                    # Send the true EOF signal ONLY when the loop is actually finished
                    try:
                        await websocket.send(json.dumps({"text": ""}))
                    except websockets.exceptions.ConnectionClosed:
                        pass

            async def receive_audio():
                while is_agent_speaking:
                    try:
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        if "error" in data:
                            print(f"\n❌ [ElevenLabs API Error]: {data['error']}")
                            break

                        if data.get("audio"):
                            print(f"\n[Debug] Got {len(data['audio'])} b64 chars of audio")
                            
                            raw_pcm = base64.b64decode(data["audio"])
                            
                            if len(raw_pcm) % 2 != 0:
                                raw_pcm += b'\x00'
                                
                            await asyncio.to_thread(speaker_stream.write, raw_pcm)
                            
                        if data.get("isFinal"):
                            break
                            
                    except websockets.exceptions.ConnectionClosed as cc:
                        print(f"\n❌ [ElevenLabs WS Closed]: Code {cc.code} - {cc.reason}")
                        break
                    except Exception as e:
                        print(f"\n❌ [Audio Playback Error]: {e}")
                        break

            await asyncio.gather(send_text(), receive_audio())
            
    except Exception as e:
        print(f"\n❌ [ElevenLabs Connection Error]: {e}")

async def handle_llm_and_tts(user_text):
    global is_agent_speaking, speech_start_time, conversation_history
    
    try:
        is_agent_speaking = True
        speech_start_time = time.time() 
        
        print(f"\n[You]: {user_text}")
        print("[Agent]: ", end="", flush=True)

        conversation_history.append({"role": "user", "content": user_text})

        # YOUR requested Groq model
        stream = await llm_client.chat.completions.create(
            model="openai/gpt-oss-20b", 
            messages=conversation_history,
            stream=True,
            max_tokens=150,
            reasoning_effort="low"
        )

        agent_response = ""

        async def token_generator():
            nonlocal agent_response
            async for chunk in stream:
                if not is_agent_speaking: 
                    break
                
                # THE FIX: Only yield actual text. Ignore None and empty strings so we don't accidentally send EOF to ElevenLabs.
                token = chunk.choices[0].delta.content
                if token: 
                    agent_response += token
                    print(token, end="", flush=True)
                    yield token
            print("\n")

        await stream_to_elevenlabs(token_generator())
        
        if agent_response:
            conversation_history.append({"role": "assistant", "content": agent_response})
            
    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f"\n❌ [LLM API Error]: {e}")
    finally:
        is_agent_speaking = False

def on_message(self, result, **kwargs):
    global is_agent_speaking, active_task

    transcript = result.channel.alternatives[0].transcript
    elapsed = time.time() - speech_start_time

    if not result.is_final and transcript and is_agent_speaking:
        if elapsed > 1.2 and len(transcript.strip()) > 5:
            is_agent_speaking = False
            print("\n[Interrupted!]")

    if result.is_final and transcript:
        if active_task and not active_task.done():
            active_task.cancel()
            
        if main_loop is not None:
            active_task = asyncio.run_coroutine_threadsafe(
                handle_llm_and_tts(transcript), main_loop
            )

async def main():
    global main_loop
    main_loop = asyncio.get_running_loop()

    dg_connection = dg_client.listen.websocket.v("1")
    dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)

    options = LiveOptions(
        model="nova-2",
        language="en-US",
        smart_format=True,
        endpointing=200,
        interim_results=True,
        encoding="linear16",      
        sample_rate=INPUT_RATE,   
        channels=CHANNELS,        
    )

    if not dg_connection.start(options):
        print("Failed to connect to Deepgram.")
        return

    print("\n🟢 Pipeline Active! Speak into your microphone (or speak over the agent to interrupt)...")
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