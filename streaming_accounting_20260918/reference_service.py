"""Real cached-model reference service for protocol testing when vLLM cannot start.

Uses Transformers eager inference and serializes generation. It is NOT vLLM,
continuous batching, a production serving engine, or a replacement GPU result.
"""
import argparse
import asyncio
import json
import time

from aiohttp import web
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def create_app(dtype, device):
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
                dtype=torch.float16 if dtype == "half" else torch.float32,
                attn_implementation="eager").to(device).eval()
    semaphore = asyncio.Semaphore(1)

    async def models(request):
        return web.json_response({"object": "list", "data": [{"id": MODEL, "object": "model",
              "owned_by": "local_transformers_reference", "max_model_len": 512}]})

    def error(message, status):
        return web.json_response({"error": {"message": message, "type": "invalid_request_error"}}, status=status)

    async def completions(request):
        try:
            body = await request.json()
            if body.get("model") != MODEL:
                return error("Model not found", 404)
            max_tokens = body.get("max_tokens", 32)
            if not isinstance(max_tokens, int) or max_tokens < 1:
                return error("max_tokens must be a positive integer", 400)
            if body.get("stream") is not True or body.get("temperature", 0) != 0:
                return error("Reference service supports only streaming greedy generation", 400)
            ids = tokenizer.apply_chat_template(body["messages"], tokenize=True,
                                                add_generation_prompt=True, return_dict=False)
            if len(ids) + max_tokens > 512:
                return error("Prompt and generation exceed 512-token context", 400)
        except (ValueError, KeyError, TypeError) as exc:
            return error(str(exc), 400)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await response.prepare(request)
        async def send(obj):
            await response.write(("data: " + (obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)) + "\n\n").encode())
        cache = None
        current = torch.tensor([ids], device=device)
        generated = []
        old_text = ""
        finish_reason = "length"
        try:
            async with semaphore:
                await send({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "token_ids": []}]})
                for index in range(max_tokens):
                    if request.transport is None or request.transport.is_closing():
                        return response
                    def step():
                        with torch.inference_mode():
                            output = model(input_ids=current, past_key_values=cache, use_cache=True)
                            token = int(output.logits[0, -1].argmax().item())
                            return token, output.past_key_values
                    token, cache = await asyncio.to_thread(step)
                    generated.append(token)
                    text = tokenizer.decode(generated, skip_special_tokens=True)
                    # This tiny service is for benchmark verification, not general detokenization.
                    # Replacement characters can be revised across partial UTF-8 token boundaries;
                    # buffer trailing replacement characters before emitting text.
                    stable_text = text.rstrip("\ufffd")
                    delta = stable_text[len(old_text):] if stable_text.startswith(old_text) else ""
                    old_text = stable_text
                    await send({"choices": [{"index": 0, "delta": {"content": delta}, "token_ids": [token]}]})
                    if token == tokenizer.eos_token_id and not body.get("ignore_eos", False):
                        finish_reason = "stop"
                        break
                    current = torch.tensor([[token]], device=device)
                await send({"choices": [{"index": 0, "delta": {}, "token_ids": [], "finish_reason": finish_reason}]})
                await send({"choices": [], "usage": {"prompt_tokens": len(ids), "completion_tokens": len(generated),
                                                      "total_tokens": len(ids) + len(generated)}})
                await send("[DONE]")
                await response.write_eof()
        except (ConnectionResetError, asyncio.CancelledError):
            # Client timeout/cancel: stop generating after the current forward pass.
            pass
        finally:
            del cache
        return response

    app = web.Application()
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", completions)
    print(f"REFERENCE_READY {MODEL} dtype={dtype} device={device} serialized=True", flush=True)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["half", "float"], default="half")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--port", type=int, default=8018)
    args = parser.parse_args()
    web.run_app(create_app(args.dtype, args.device), host="127.0.0.1", port=args.port)
