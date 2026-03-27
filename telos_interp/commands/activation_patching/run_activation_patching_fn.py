import json

import nnsight
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ANSWER_FORCING_PROMPT_SUFFIX = (
    '<|channel|>analysis<|message|> <|end|><|start|>assistant<|channel|>final<|message|>{\n  "action": "'
)


def get_agent_action(trajectory: dict) -> str:
    return trajectory["steps"][0]["agent_action"]


def get_full_input_tokens(trajectory: dict) -> list[int]:
    prompt_data = trajectory["prompt"]
    prefix_tokens = [token["token_id"] for token in prompt_data["prompt_prefix_tokens"]]
    prompt_suffix_tokens = [token["token_id"] for token in prompt_data["prompt_suffix_tokens"]]

    first_step = trajectory["steps"][0]
    grid_state_tokens = [token["token_id"] for token in first_step["grid_state_tokens"]]

    full_input_tokens = prefix_tokens + grid_state_tokens + prompt_suffix_tokens
    return full_input_tokens


def get_answer_forcing_input(trajectory: dict, tokenizer: AutoTokenizer) -> str:
    """Get the answer forcing input for the trajectory. This follows the Harmony format and provides empty analysis channel."""
    full_input_tokens = get_full_input_tokens(trajectory)

    full_input_string = tokenizer.decode(full_input_tokens)

    answer_forcing_input_string = full_input_string + ANSWER_FORCING_PROMPT_SUFFIX
    answer_forcing_input_ids = tokenizer.encode(answer_forcing_input_string, return_tensors="pt").to(DEVICE)
    return answer_forcing_input_ids, answer_forcing_input_string


def get_grid_start_end(trajectory: dict) -> tuple[int, int]:
    """Get the start and end positions of the grid in the trajectory."""
    grid_start = len(trajectory["prompt"]["prompt_prefix_tokens"])
    grid_end = grid_start + trajectory["steps"][0]["grid_state_n_tokens"]
    return grid_start, grid_end


def run_activation_patching():
    """Run activation patching experiments."""
    # trajectories_dir_path = Path(trajectories_dir)

    # json_files = glob(f"{trajectories_dir_path}/*.json")

    # Action LEFT
    # clean_filename = "data/trajectories_train_single_step/size9/together_ai_openai_gpt-oss-20b_size9_comp0.2_0.json"
    clean_filename = "data/trajectories_train_single_step/size7/together_ai_openai_gpt-oss-20b_size7_comp0.0_0.json"

    with open(clean_filename) as f:
        clean_trajectory = json.load(f)

    model_id = clean_trajectory["model_params"]["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    clean_action = get_agent_action(clean_trajectory)
    clean_action_id = tokenizer.encode(clean_action, return_tensors="pt").to(DEVICE)

    clean_full_input_tokens = get_full_input_tokens(clean_trajectory)
    clean_input_length = len(clean_full_input_tokens)
    grid_start, grid_end = get_grid_start_end(clean_trajectory)
    # patching_begin = grid_start
    # patching_end = grid_end
    patching_begin = clean_input_length - 3
    patching_end = clean_input_length

    # We patch at the very end of clean input, which is <|end|><|start|>assistant tokens

    # Load model first
    model = AutoModelForCausalLM.from_pretrained(model_id).to(DEVICE)

    answer_forcing_input_ids, answer_forcing_input_string = get_answer_forcing_input(clean_trajectory, tokenizer)
    print(f"Answer forcing input: {answer_forcing_input_string}")

    # Do one forward pass to see which actions the model thinks are most likely.
    outputs = model(answer_forcing_input_ids, output_logits=True)

    logits = outputs.logits  # tensor of shape (batch_size, max_new_tokens, vocab_size)
    logits = logits[0][-1]  # (vocab_size,)
    probabilities = torch.softmax(logits, dim=-1)
    top_indices = torch.topk(probabilities, 5).indices
    print(f"Top 5 indices: {top_indices}")
    # detokenize
    for idx in top_indices.tolist():
        token = tokenizer.decode(idx)
        print(f"Token: {token}")
        print(f"Probability: {probabilities[idx]}")
        print("--------------------------------")

    clean_top_token_id = top_indices[0].item()

    clean_action_probability = probabilities[clean_action_id].item()

    print(f"Clean action probability: {clean_action_probability}")

    # # generate a full answer, without forcing
    # clean_full_input_ids = torch.tensor([clean_full_input_tokens]).to(DEVICE)
    # input_len = clean_full_input_ids.shape[1]

    # clean_outputs = model.generate(clean_full_input_ids, max_new_tokens=1024)
    # clean_answer = tokenizer.decode(clean_outputs[0,input_len:], skip_special_tokens=True)
    # print(f"Clean answer: {clean_answer}")

    # # Action RIGHT
    corrupted_filename = (
        "data/trajectories_train_single_step/size7/together_ai_openai_gpt-oss-20b_size7_comp0.0_34.json"
    )

    with open(corrupted_filename) as f:
        corrupted_trajectory = json.load(f)


    corrupted_answer_forcing_input_ids, corrupted_answer_forcing_input_string = get_answer_forcing_input(
        corrupted_trajectory, tokenizer
    )
    print(f"Corrupted answer forcing input: {corrupted_answer_forcing_input_string}")

    corrupted_outputs = model(corrupted_answer_forcing_input_ids, output_logits=True)
    corrupted_logits = corrupted_outputs.logits[0][-1]  # (vocab_size,)
    corrupted_probabilities = torch.softmax(corrupted_logits, dim=-1)
    corrupted_top_indices = torch.topk(corrupted_probabilities, 5).indices
    print(f"Top 5 indices: {corrupted_top_indices}")
    # detokenize
    for idx in corrupted_top_indices.tolist():
        token = tokenizer.decode(idx)
        print(f"Token: {token}")
        print(f"Probability: {corrupted_probabilities[idx]}")
        print("--------------------------------")
    corrupted_top_token_id = corrupted_top_indices[0].item()

    # Start the patching experiment
    patched_tokens_in_clean = answer_forcing_input_ids[0][patching_begin:patching_end]
    print(f"Patched tokens in clean: {patched_tokens_in_clean}")
    print(f"Decoded patched tokens in clean: {tokenizer.decode(patched_tokens_in_clean)}")

    clean_prompt = answer_forcing_input_string
    corrupted_prompt = corrupted_answer_forcing_input_string

    correct_index = clean_top_token_id
    incorrect_index = corrupted_top_token_id

    nnsight_model = nnsight.LanguageModel(model, tokenizer=tokenizer, dispatch=True)

    N_LAYERS = model.config.num_hidden_layers

    # Clean run
    clean_hs = []
    with nnsight_model.trace() as tracer:
        with tracer.invoke(clean_prompt):

            # Get hidden states of all layers in the network.
            # We index the output at 0 because it's a tuple where the first index is the hidden state.
            for layer_idx in range(N_LAYERS):
                clean_hs.append(nnsight_model.model.layers[layer_idx].output[0].save())

            # Get logits from the lm_head.
            clean_logits = nnsight_model.lm_head.output

            clean_probabilities = torch.softmax(clean_logits, dim=-1)

            clean_correct_probability = clean_probabilities[0, -1, correct_index].save()
            clean_incorrect_probability = clean_probabilities[0, -1, incorrect_index].save()

    clean_prompt_len = answer_forcing_input_ids.shape[1]
    print(f"Clean prompt length: {clean_prompt_len}")
    print(f"Shape of each clean hidden state: {clean_hs[0].shape}")

    print(f"Clean correct probability: {clean_correct_probability}")
    print(f"Clean incorrect probability: {clean_incorrect_probability}")

    # Corrupted run
    with nnsight_model.trace(corrupted_prompt) as tracer:
        corrupted_logits = nnsight_model.lm_head.output

        corrupted_probabilities = torch.softmax(corrupted_logits, dim=-1)
        corrupted_correct_probability = corrupted_probabilities[0, -1, correct_index].save()
        corrupted_incorrect_probability = corrupted_probabilities[0, -1, incorrect_index].save()

    print(f"Corrupted correct probability: {corrupted_correct_probability}")
    print(f"Corrupted incorrect probability: {corrupted_incorrect_probability}")

    # Activation Patching Intervention

    # We do just one forward pass and patch at each patching token for all layers.
    with nnsight_model.trace(corrupted_prompt) as tracer:
        # Apply the patch from the clean hidden states to the corrupted hidden states.
        # model.transformer.h[layer_idx].output[0][:, token_idx] = clean_hs[layer_idx][:,token_idx,:]

        for layer_idx in range(N_LAYERS):
            clean_patched_activations = clean_hs[layer_idx][patching_begin:patching_end, :]
            print(f"shape of model output: {nnsight_model.model.layers[layer_idx].output[0].shape}")

            nnsight_model.model.layers[layer_idx].output[0][patching_begin:patching_end, :] = clean_patched_activations

        patched_logits = nnsight_model.lm_head.output
        patched_probabilities = torch.softmax(patched_logits, dim=-1)
        patched_correct_probability = patched_probabilities[0, -1, correct_index].save()
        patched_incorrect_probability = patched_probabilities[0, -1, incorrect_index].save()

    print(f"Patched correct probability: {patched_correct_probability}")
    print(f"Patched incorrect probability: {patched_incorrect_probability}")
