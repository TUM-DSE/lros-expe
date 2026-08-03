import sys 
import json
import re

OUTPUT_FILE = "shareGPT_clean.txt"

def clean_file(filepath: str):
    f = open(filepath)
    f_out = open(OUTPUT_FILE, 'w+')
    data = json.load(f)

    for item in data:
        try:
            prompt = item.get('conversations')[0]
        except Exception as e:
            continue

        if prompt.get('from') == 'human':
            prompt = prompt.get('value')
            prompt = prompt.replace('\n', ' ')
            if len(prompt) > 1024 or len(prompt) < 40:
                continue
            if prompt.isascii() and "{" not in prompt and "}" not in prompt and "\"" not in prompt and "\'" not in prompt:
                prompt = bytes(prompt, 'utf-8').decode('utf-8', 'ignore')
                f_out.write(prompt + '\n')

    f.close()
    f_out.close()

if __name__ == "__main__":
    if len(sys.argv) - 1 != 1:
        print(f"Usage: python {sys.argv[0]} <shareGPT_file>")
        exit(0)
    clean_file(sys.argv[1])
