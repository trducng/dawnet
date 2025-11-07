"""Utility methods to play around with tokens"""
import random
from textwrap import dedent

import torch
from IPython.display import display, HTML
import html


class Tokens:
    def __init__(self, tensor, tokenizer):
        self.tensor = tensor
        self.tokenizer = tokenizer
        self._common_functional_tokens = ["<think>", "</think>"]

    def view_nb(self):
        """
        Display decoded tokens with interactive hover tooltips.

        Args:
            tensor: PyTorch tensor of shape (n,) containing token indices
            tokenizer: Tokenizer object with a decode method

        When hovering over tokens, shows:
            - Token index (the actual token ID)
            - Position in tensor (0-indexed position)
        """
        tensor = self.tensor[0]
        # Convert tensor to list if needed
        if isinstance(tensor, torch.Tensor):
            token_ids = tensor.cpu().tolist()
        else:
            token_ids = list(tensor)

        # Decode each token individually
        tokens = []
        for token_id in token_ids:
            try:
                # Try to convert token to string directly first
                if hasattr(self.tokenizer, 'convert_ids_to_tokens'):
                    token_str = self.tokenizer.convert_ids_to_tokens(token_id)
                    # Handle special tokens that might be None
                    if token_str is None:
                        token_str = self.tokenizer.decode([token_id])
                else:
                    token_str = self.tokenizer.decode([token_id])

                # Clean up common tokenizer artifacts for better readability
                if token_str.startswith('Ġ'):  # GPT-2/RoBERTa space prefix
                    token_str = ' ' + token_str[1:]
                elif token_str.startswith('▁'):  # SentencePiece space prefix (T5, etc.)
                    token_str = ' ' + token_str[1:]

                tokens.append(token_str)
            except:
                # Fallback if decode fails
                tokens.append(f"[{token_id}]")

        # Build HTML with hover tooltips
        html_parts = [dedent("""
        <style>
            .token-container {
                font-family: 'Courier New', monospace;
                font-size: 14px;
                line-height: 1.6;
                padding: 50px;
                background-color: #ffffff;
                border: 1px solid #ddd;
                border-radius: 8px;
                max-width: 100%;
                overflow-wrap: break-word;
                white-space: pre-wrap;
                color: #000000;
            }
            .token {
                position: relative;
                display: inline-block;
                cursor: pointer;
                padding: 1px;
                margin: 0;
                border-radius: 3px;
                transition: background-color 0.2s;
                color: #000000;
                white-space: pre;
                vertical-align: top;
                border: 1px solid transparent;
                font-size: inherit;
                line-height: inherit;
            }
            .token:hover {
                background-color: #ffeb3b;
                border-color: #fbc02d;
            }
            .token-tooltip {
                visibility: hidden;
                position: absolute;
                background-color: #333;
                color: #fff;
                text-align: left;
                padding: 8px 12px;
                border-radius: 6px;
                z-index: 1000;
                top: 125%;
                left: 50%;
                transform: translateX(-50%);
                white-space: nowrap;
                opacity: 0;
                transition: opacity 0.3s;
                font-size: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            }
            .token-tooltip::after {
                content: "";
                position: absolute;
                bottom: 100%;
                left: 50%;
                margin-left: -5px;
                border-width: 5px;
                border-style: solid;
                border-color: transparent transparent #333 transparent;
            }
            .token:hover .token-tooltip {
                visibility: visible;
                opacity: 1;
            }
            .token-info {
                display: block;
                margin: 2px 0;
            }
        </style>
        <div class="token-container">
        """).strip()]

        # Create span for each token with tooltip - no newlines between spans
        for position, (token_id, token_text) in enumerate(zip(token_ids, tokens)):
            # Escape HTML in token text
            escaped_text = html.escape(token_text)

            # Create tooltip content
            tooltip_html = f'<span class="token-tooltip"><span class="token-info"><strong>Position:</strong> {position}</span><span class="token-info"><strong>Token ID:</strong> {token_id}</span></span>'

            # Create token span - all in one line
            token_html = f'<span class="token">{escaped_text}{tooltip_html}</span>'
            html_parts.append(token_html)

        html_parts.append("</div>")

        # Display the HTML
        full_html = "".join(html_parts)
        display(HTML(full_html))

    def drop_tokens(self, indices, keep: list | None = None):
        """Drop tokens with indices `indices`
        """
        to_keep_token_id = set(keep or [])
        to_drop = set(indices)
        out = []
        for idx in range(self.tensor.shape[1]):
            if (
                (idx in to_drop) and
                (self.tensor[0,idx].cpu().item() not in to_keep_token_id)
            ):
                continue
            out.append(self.tensor[0,idx])
        return Tokens(
            tensor=torch.tensor(
                data=[out], dtype=self.tensor.dtype, device=self.tensor.device
            ),
            tokenizer=self.tokenizer
        )

    def drop_tokens_randomly(self, pdrop, indices=None, keep: list | None = None, truncate: bool = True):
        """Drop the tokens that contain in the indices randomly
        """
        if indices is None:
            if truncate:
                print("no truncation if not specify `indices`")
            to_drop = set(range(self.tensor.shape[1]))
            end_idx = self.tensor.shape[1]
        else:
            to_drop = set(indices)
            end_idx = min(max(indices), self.tensor.shape[1])
        to_keep_token_id = set(keep or [])
        out = []
        for idx in range(end_idx):
            if (
                (idx in to_drop) and
                (self.tensor[0,idx].cpu().item() not in to_keep_token_id)
            ):
                if random.random() < pdrop:
                    continue
            out.append(self.tensor[0,idx])
        return Tokens(
            tensor=torch.tensor(
                data=[out], dtype=self.tensor.dtype, device=self.tensor.device
            ),
            tokenizer=self.tokenizer
        )

    def recommend_keep_tokens(self):
        token_ids = []
        for tok in self._common_functional_tokens:
            tok_id = self.tokenizer.encode(tok) 
            if len(tok_id) == 1:
                token_ids.append(tok_id[0])
        token_ids += self.tokenizer.all_special_ids
        return token_ids

    def __str__(self):
        return self.tokenizer.decode(self.tensor[0])
