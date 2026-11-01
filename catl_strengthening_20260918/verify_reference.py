"""Cross-check the custom loop against Transformers' independent generate API."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    args = p.parse_args()
    directory = Path(args.run)
    torch.set_num_threads(4)
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=True,
                dtype=torch.float16, attn_implementation="eager").to("cuda").eval()
    data = json.loads((directory / "fp16_quality.json").read_text(encoding="utf-8"))
    selected = {"arith_1", "lookup_1", "rule_1", "logic_3"}
    results = []
    for case in data["results"]:
        if case["id"] not in selected:
            continue
        encoded = tokenizer.apply_chat_template([{"role": "user", "content": case["prompt"]}],
                     tokenize=True, add_generation_prompt=True, return_dict=False, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            generated = model.generate(encoded, attention_mask=torch.ones_like(encoded),
                          max_new_tokens=48, do_sample=False, use_cache=True,
                          repetition_penalty=1.0,
                          eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
        ids = generated[0, encoded.shape[1]:].tolist()
        results.append({"id": case["id"], "generate_token_ids": ids,
                        "reference_token_ids": case["record"]["token_ids"],
                        "exact_token_match": ids == case["record"]["token_ids"]})
    result = {"kind": "independent_generation_api_equivalence", "dtype": "fp16", "cases": results,
              "repetition_penalty": 1.0,
              "passed": sum(r["exact_token_match"] for r in results), "total": len(results)}
    (directory / "generate_equivalence_repetition1.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "total": result["total"]}))
    if result["passed"] != result["total"] or result["total"] != 4:
        raise RuntimeError("Reference loop differs from Transformers.generate; inspect before reporting")


if __name__ == "__main__":
    main()
