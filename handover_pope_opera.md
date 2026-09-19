# Hand-off Context: OPERA Hallucination Mitigation Project

## 1. Environment & Workflow Setup
- **Local Workspace (Antigravity Agent's Domain):** MacOS, `/Users/nguyenmy/GitHub/OPERA/`
- **Remote Server (Execution Domain):** Ubuntu Server (`nvidia-lab@nvidia-lab`), Path: `~/ai4life/phuongnh/vlm-truth/4. implement method/OPERA/`
- **Workflow:** The Agent writes/edits code in the Local Workspace. The User manually syncs these files to the Remote Server and runs the commands via terminal.
- **Conda Env (Remote):** `vlm_truth_py313`
- **Hardware Constraints:** H100 80GB GPU, but highly shared. VRAM is scarce. The model requires ~18GB VRAM because it MUST run in `eager` attention mode to expose attention weights.

## 2. Codebase Architecture & Current Status
- **Core Algorithm:** `opera_core/opera_decoding.py` contains `OverTrustPenaltyLogitsProcessor`. It uses PyTorch forward hooks (`register_opera_attention_hooks`) to intercept attention matrices from the LLaVA model during the HF `generate()` loop.
- **Model Wrapper:** `models/llava_wrapper.py` initializes LLaVA-1.5-7B with `attn_implementation="eager"`. It properly formats prompts for LLaVA (`USER: <image>\n{prompt}\nASSISTANT:`).
- **Paths:** Managed centrally in `data_paths.yaml`. The LLaVA model is loaded locally from `/data/ce/model/llava-v1.5-7b` due to disk space issues on the server.
- **Milestone Reached:** Successfully ran the BEAF benchmark. **Critical finding during BEAF:** The generated output JSON must use the key `"answer"` instead of `"pred"` for the metric scripts to parse them correctly.

## 3. Next Goal: POPE Benchmark
The User needs to run the **POPE** benchmark for **3 splits** (`adversarial`, `random`, `popular`) using the **OPERA** intervention (`--use_opera` flag).

### Tasks for the Next Agent:
1. **Review `run_pope.py`:** Check `benchmarks/pope/run_pope.py` in the local workspace. Ensure that the generated JSON format correctly outputs the key `"answer"` (or whatever the official POPE evaluation script expects) instead of `"pred"`.
2. **Bash Script Generation:** Write a `.sh` script that automates the execution of `run_pope.py` for all 3 splits consecutively to save the user from running them manually.
3. **Execution Instructions:** Provide the user with the exact commands to run the POPE benchmark and the corresponding evaluation script (`pope_eval.py` or similar).

## 4. Antigravity Remote Access Note
*Note on Remote Access:* Antigravity operates natively on the local filesystem. To allow Antigravity to interact directly with the remote server, the user can either:
1. Run `ssh nvidia-lab@<ip_address> "<command>"` via the agent's bash tools (requires SSH keys without password).
2. Mount the remote server directory to the Mac using `sshfs`.
3. Connect the Antigravity IDE directly to the remote server via VS Code Remote-SSH (if supported).
