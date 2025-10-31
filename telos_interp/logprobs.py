import math

from matplotlib import pyplot as plt


def extract_action_probabilities(data):
    """
    Extract action probabilities from LLM output containing logprobs.

    This function processes the logprobs from an LLM that generates navigation actions
    (UP, DOWN, LEFT, RIGHT) and returns the probability distribution for each token
    in the generation.

    Args:
        data (dict): A dictionary with the following structure:
            {
                "metadata": {
                    "logprobs": [
                        {
                            "token": str,
                            "logprob": float,
                            "top_logprobs": [
                                {
                                    "token": str,
                                    "logprob": float
                                },
                                ...
                            ]
                        },
                        ...
                    ]
                }
            }

    Returns:
        list: A list of dictionaries, one per generated token. Each dictionary contains:
            {
                "UP": float,     # Probability of UP action
                "DOWN": float,   # Probability of DOWN action
                "LEFT": float,   # Probability of LEFT action
                "RIGHT": float   # Probability of RIGHT action
            }

    Notes:
        - Probabilities are computed as exp(logprob)
        - Tokens representing the same action but written differently (e.g., "UP", "up",
          "Up", "UP ") are merged by summing their probabilities
        - Actions not present in the top logprobs have probability 0
        - The function normalizes tokens by stripping whitespace and converting to uppercase

    Example:
        >>> data = {
        ...     "metadata": {
        ...         "logprobs": [
        ...             {
        ...                 "token": "RIGHT",
        ...                 "top_logprobs": [
        ...                     {"token": "RIGHT", "logprob": 0.0},
        ...                     {"token": "right", "logprob": -2.3},
        ...                     {"token": " RIGHT", "logprob": -3.1},
        ...                     {"token": "LEFT", "logprob": -4.5}
        ...                 ]
        ...             }
        ...         ]
        ...     }
        ... }
        >>> result = extract_action_probabilities(data)
        >>> # result[0]['RIGHT'] will be exp(0.0) + exp(-2.3) + exp(-3.1)
    """
    # Define valid actions
    ACTIONS = {"UP", "DOWN", "LEFT", "RIGHT"}

    # Get logprobs from metadata
    logprobs_list = data.get("metadata", {}).get("logprobs", [])

    result = []

    # Process each token in the generation
    for idx, token_data in enumerate(logprobs_list):
        # Get top logprobs for this token
        top_logprobs = token_data.get("top_logprobs", [])

        # Initialize action probabilities to 0
        action_probs = {
            "token": token_data.get("token", ""),
            "token_index": idx,
            "UP": 0.0,
            "DOWN": 0.0,
            "LEFT": 0.0,
            "RIGHT": 0.0,
        }

        # Process each logprob entry in the top predictions
        for logprob_entry in top_logprobs:
            token = logprob_entry.get("token", "")
            logprob = logprob_entry.get("logprob", float("-inf"))

            # Normalize token: strip whitespace and convert to uppercase
            normalized_token = token.strip().upper()

            # Check if normalized token matches an action
            if normalized_token in ACTIONS:
                # Convert logprob to probability
                prob = math.exp(logprob)
                # Add to the corresponding action (merge probabilities for same action)
                action_probs[normalized_token] += prob

        result.append(action_probs)

    return result


def plot_action_probabilities(
    action_probs, figsize=(14, 6), save_path=None, show_tokens=True, show_text=True, text_fontsize=12
):
    """
    Plot the progression of action probabilities throughout the generation.

    Args:
        action_probs (list): List of dictionaries with keys: 'token', 'token_index',
                            'UP', 'DOWN', 'LEFT', 'RIGHT'
        figsize (tuple): Figure size (width, height) in inches
        save_path (str, optional): Path to save the figure. If None, displays the plot.
        show_tokens (bool): Whether to show token labels on x-axis for non-zero probabilities
        show_text (bool): Whether to show the generated text with color-coded tokens
        text_fontsize (int): Font size for the generated text display

    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    # Extract data
    token_indices = [item["token_index"] for item in action_probs]
    up_probs = [item["UP"] for item in action_probs]
    down_probs = [item["DOWN"] for item in action_probs]
    left_probs = [item["LEFT"] for item in action_probs]
    right_probs = [item["RIGHT"] for item in action_probs]
    tokens = [item["token"] for item in action_probs]

    # Define action colors (matching the plot lines)
    action_colors = {
        "UP": "#2E86AB",
        "DOWN": "#A23B72",
        "LEFT": "#F18F01",
        "RIGHT": "#C73E1D",
        "NONE": "#666666",  # Gray for tokens with no action
    }

    # Create figure
    if show_text:
        fig = plt.figure(figsize=figsize)
        # Leave space at bottom for text (30% of figure height for better spacing)
        ax = fig.add_axes([0.08, 0.38, 0.88, 0.58])
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot lines for each action
    ax.plot(
        token_indices, up_probs, "o-", label="UP", color=action_colors["UP"], linewidth=2.5, markersize=5, alpha=0.9
    )
    ax.plot(
        token_indices,
        down_probs,
        "s-",
        label="DOWN",
        color=action_colors["DOWN"],
        linewidth=2.5,
        markersize=5,
        alpha=0.9,
    )
    ax.plot(
        token_indices,
        left_probs,
        "^-",
        label="LEFT",
        color=action_colors["LEFT"],
        linewidth=2.5,
        markersize=5,
        alpha=0.9,
    )
    ax.plot(
        token_indices,
        right_probs,
        "d-",
        label="RIGHT",
        color=action_colors["RIGHT"],
        linewidth=2.5,
        markersize=5,
        alpha=0.9,
    )

    # Highlight tokens with significant action probabilities
    threshold = 0.01
    significant_indices = []
    for i, idx in enumerate(token_indices):
        max_prob = max(up_probs[i], down_probs[i], left_probs[i], right_probs[i])
        if max_prob > threshold:
            significant_indices.append(i)
            ax.axvline(x=idx, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

    # Annotate significant tokens
    if show_tokens and significant_indices:
        for i in significant_indices:
            idx = token_indices[i]
            max_prob = max(up_probs[i], down_probs[i], left_probs[i], right_probs[i])
            token_text = tokens[i].strip()

            if len(token_text) > 15:
                token_text = token_text[:12] + "..."

            ax.annotate(
                token_text,
                xy=(idx, max_prob),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3, edgecolor="none"),
                rotation=45,
            )

    # Customize plot
    ax.set_xlabel("Token Index", fontsize=13, fontweight="bold")
    ax.set_ylabel("Probability", fontsize=13, fontweight="bold")
    ax.set_title("Action Probabilities Throughout Generation", fontsize=15, fontweight="bold", pad=15)
    ax.legend(loc="upper left", frameon=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)
    ax.set_ylim(bottom=-0.02, top=1.05)
    ax.set_xlim(left=-1, right=max(token_indices) + 1)

    # Add generated text with color-coded highlights
    if show_text:
        # Determine dominant action for each token
        token_info = []
        for i in range(len(tokens)):
            probs = {"UP": up_probs[i], "DOWN": down_probs[i], "LEFT": left_probs[i], "RIGHT": right_probs[i]}
            max_prob = max(probs.values())

            if max_prob > threshold:
                dominant_action = max(probs, key=probs.get)
                token_info.append({"text": tokens[i], "action": dominant_action, "prob": max_prob, "has_action": True})
            else:
                token_info.append({"text": tokens[i], "action": "NONE", "prob": 0.0, "has_action": False})

        # Build text with inline formatting
        # Group consecutive tokens with same action/no-action
        text_parts = []
        current_part = {"tokens": [], "action": None, "has_action": False}

        for info in token_info:
            if info["has_action"]:
                # Start new part for each action token
                if current_part["tokens"]:
                    text_parts.append(current_part)
                text_parts.append(
                    {"tokens": [info["text"]], "action": info["action"], "has_action": True, "prob": info["prob"]}
                )
                current_part = {"tokens": [], "action": None, "has_action": False}
            # Group non-action tokens together
            elif info["action"] == current_part["action"] or current_part["action"] is None:
                current_part["tokens"].append(info["text"])
                current_part["action"] = info["action"]
            else:
                if current_part["tokens"]:
                    text_parts.append(current_part)
                current_part = {"tokens": [info["text"]], "action": info["action"], "has_action": False}

        if current_part["tokens"]:
            text_parts.append(current_part)

        # Render text parts below the plot
        text_area_top = 0.33  # Top of text area (just below plot with margin)
        y_pos = text_area_top - 0.06  # Start position for text content
        x_pos = 0.05
        max_width = 0.9
        line_height = 0.05  # Increased for better spacing

        # Render each part
        min_y = 0.02  # Minimum y position to avoid going off figure

        # First, render all text objects to get their actual bounding boxes
        text_objects = []
        for part in text_parts:
            part_text = "".join(part["tokens"])

            # Skip empty parts or parts that are only whitespace/newlines
            if not part_text.strip():
                continue

            # Replace newlines with spaces to avoid rendering issues
            part_text = part_text.replace("\n", "").replace("\r", "")

            # Render based on whether it has action
            plot_kwargs = {}
            if part["has_action"]:
                # Highlighted action token with padding to prevent overlap
                color = action_colors[part["action"]]
                bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.85, edgecolor="white", linewidth=1.5)
                plot_kwargs = {"bbox": bbox}
                part_text = part_text.strip()
                text_obj = fig.text(
                    0, 0,  # Temporary position
                    " ",
                    fontsize=text_fontsize,
                    color="white",
                    va="center",
                    family="monospace",
                )
                text_objects.append((text_obj, False))

            # Create text object (not yet positioned properly)
            text_obj = fig.text(
                0, 0,  # Temporary position
                part_text,
                fontsize=text_fontsize,
                color="white" if part["has_action"] else action_colors["NONE"],
                va="center",
                family="monospace",
                **plot_kwargs
            )
            text_objects.append((text_obj, part["has_action"]))

        # Force a draw to compute bounding boxes
        renderer = fig.canvas.get_renderer()

        # Now position the text objects properly using their actual widths
        x_pos = 0.05
        for i, (text_obj, has_action) in enumerate(text_objects):
            # Get the bounding box in figure coordinates
            bbox = text_obj.get_window_extent(renderer=renderer)
            bbox_fig = bbox.transformed(fig.transFigure.inverted())
            text_width = bbox_fig.width

            # Check for line wrap
            if x_pos + text_width > max_width and x_pos > 0.05:
                x_pos = 0.05
                y_pos -= line_height

                # Stop if we're too close to bottom
                if y_pos < min_y:
                    # Remove this and remaining text objects
                    for j in range(i, len(text_objects)):
                        text_objects[j][0].remove()
                    # Add ellipsis to indicate truncation
                    fig.text(x_pos, y_pos + line_height, "...", fontsize=text_fontsize, color="#999999", va="center")
                    break

            # Position the text object
            text_obj.set_position((x_pos, y_pos))

            # Move x position for next text
            # Add extra padding for action tokens to prevent overlap
            gap = 0.006 if has_action else 0.003
            x_pos += text_width + gap
    else:
        plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    return fig


# Example usage
if __name__ == "__main__":
    import json

    # Example: Load data and extract probabilities
    with open("data/single_cell_results_ex.json") as f:
        data = json.load(f)

    action_probs = extract_action_probabilities(data)
    plot_action_probabilities(action_probs, save_path="action_probabilities.png")
