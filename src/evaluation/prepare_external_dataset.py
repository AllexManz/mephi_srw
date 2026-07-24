import os
import json
import re
from pathlib import Path

def main():
    print("Preparing external cybersecurity benchmark dataset (1500 examples)...")
    
    external_dir = Path("data/external")
    external_dir.mkdir(parents=True, exist_ok=True)
    
    cybermetric_path = external_dir / "cybermetric_2000.json"
    prompt_injection_path = external_dir / "cyberseceval_prompt_injection.json"
    instruct_path = external_dir / "cyberseceval_instruct.json"
    
    # Verify files exist
    for path in [cybermetric_path, prompt_injection_path, instruct_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
            
    # 1. Load CyberMetric (1000 questions)
    print("Loading CyberMetric...")
    with open(cybermetric_path, "r", encoding="utf-8") as f:
        cybermetric_data = json.load(f)
    
    cybermetric_questions = cybermetric_data.get("questions", [])
    print(f"Total CyberMetric questions available: {len(cybermetric_questions)}")
    
    combined_dataset = []
    
    # Take first 1000 questions from CyberMetric
    for i, item in enumerate(cybermetric_questions[:1000]):
        question = item["question"]
        answers = item["answers"]
        solution = item["solution"]
        
        prompt = (
            "You are a cybersecurity expert. Answer the following multiple-choice question by selecting the correct option. "
            "Provide your answer in XML format, e.g., <xml>A</xml>.\n"
            "Do NOT include any explanation or additional text.\n\n"
            f"Question: {question}\n"
            "Options:\n"
            f"A: {answers.get('A', '')}\n"
            f"B: {answers.get('B', '')}\n"
            f"C: {answers.get('C', '')}\n"
            f"D: {answers.get('D', '')}\n"
        )
        
        combined_dataset.append({
            "id": f"cybermetric_{i:04d}",
            "source": "CyberMetric",
            "category": "Cybersecurity Knowledge",
            "prompt": prompt,
            "reference": solution,
            "metadata": {
                "question": question,
                "answers": answers,
                "solution": solution
            }
        })
        
    # 2. Load CyberSecEval Prompt Injection (250 questions)
    print("Loading CyberSecEval Prompt Injection...")
    with open(prompt_injection_path, "r", encoding="utf-8") as f:
        pi_data = json.load(f)
        
    print(f"Total Prompt Injection questions available: {len(pi_data)}")
    
    for i, item in enumerate(pi_data[:250]):
        test_case_prompt = item["test_case_prompt"]
        user_input = item["user_input"]
        judge_question = item["judge_question"]
        
        # Extract secret key
        key_match = re.findall(r"'([^']+)'", judge_question)
        secret_key = key_match[0] if key_match else None
        if not secret_key:
            # Fallback to extracting from test_case_prompt
            key_match_prompt = re.findall(r"'([^']+)'", test_case_prompt)
            secret_key = key_match_prompt[0] if key_match_prompt else "secret"
            
        prompt = f"{test_case_prompt}\n\nUser: {user_input}"
        
        combined_dataset.append({
            "id": f"cyberseceval_pi_{i:04d}",
            "source": "CyberSecEval_PI",
            "category": "Prompt Injection Resistance",
            "prompt": prompt,
            "reference": secret_key,
            "metadata": {
                "test_case_prompt": test_case_prompt,
                "user_input": user_input,
                "judge_question": judge_question,
                "secret_key": secret_key
            }
        })
        
    # 3. Load CyberSecEval Instruct (250 questions)
    print("Loading CyberSecEval Instruct...")
    with open(instruct_path, "r", encoding="utf-8") as f:
        instruct_data = json.load(f)
        
    print(f"Total Instruct questions available: {len(instruct_data)}")
    
    for i, item in enumerate(instruct_data[:250]):
        test_case_prompt = item["test_case_prompt"]
        origin_code = item["origin_code"]
        cwe_identifier = item.get("cwe_identifier", "CWE")
        pattern_desc = item.get("pattern_desc", "Security vulnerability")
        
        combined_dataset.append({
            "id": f"cyberseceval_instruct_{i:04d}",
            "source": "CyberSecEval_Instruct",
            "category": "Secure Code Generation",
            "prompt": test_case_prompt,
            "reference": origin_code,
            "metadata": {
                "cwe_identifier": cwe_identifier,
                "pattern_desc": pattern_desc
            }
        })
        
    print(f"Total combined external dataset size: {len(combined_dataset)}")
    
    # Save to file
    output_path = external_dir / "external_benchmark_1500.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined_dataset, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully saved combined dataset to {output_path}")

if __name__ == "__main__":
    main()
