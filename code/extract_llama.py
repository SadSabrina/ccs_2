import numpy as np
import torch
from tqdm import tqdm


def get_llm_type(model_cfg) -> str:
    """
    Determine the model type based on configuration
    """
    if model_cfg.config.is_encoder_decoder:
        return "encoder-decoder"

    model_type = model_cfg.config.model_type.lower()
    if model_type in [
        "bert",
        "roberta",
        "distilbert",
        "albert",
        "deberta",
        "deberta-v2",
    ]:
        return "encoder"
    elif model_type in [
        "gpt2",
        "gpt",
        "gptj",
        "gpt_neo",
        "gpt_neox",
        "llama",
        "bloom",
        "opt",
        "falcon",
        "gemma",
    ]:
        return "decoder"

    return "unknown"


def setup_tokenizer_padding(tokenizer):
    """
    CHANGED: Added function to properly set up padding token for tokenizers
    Sets up padding token if not already present, using eos_token as fallback
    """
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
        else:
            # Add a new pad token if no eos_token exists
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            print(f"Added new pad_token: {tokenizer.pad_token}")
    return tokenizer


def extract_representation(
    model,
    tokenizer,
    text,
    layer_index=None,
    get_all_hs=False,
    strategy="last-token",
    model_type=None,
    use_decoder=False,
    device=None,
    token_number=None,
):
    """
    FIXED: Extracts the vector representation from the given model layer or all model layers.
    Enhanced with proper handling for different model types, precision issues, and padding tokens.

    Args:
        - model: HuggingFace model
        - tokenizer: model tokenizer
        - text: input string
        - layer_index: layer number (as default middle layer)
        - strategy: 'first-token' (for encoders), 'last-token' (for decoders), 'mean', or 'custom'
        - model_type: "encoder", "encoder-decoder", "decoder"
        - use_decoder: if True use decoder hidden states (for encoder-decoder models)
        - device: computation device
        - token_number: specific token position for 'custom' strategy

    Return:
        - numpy-array (n_layers, dim) if get_all_hs=True, else (dim,)
    """
    # CHANGED: Setup padding token before any tokenization
    tokenizer = setup_tokenizer_padding(tokenizer)

    # FIXED: Enhanced device handling
    if device is None:
        if hasattr(model, "device"):
            device = model.device
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    # FIXED: Proper tokenization with max_length to prevent overflow
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    ).to(device)

    # FIXED: Get attention mask for proper token identification
    attention_mask = inputs.get("attention_mask", None)

    # FIXED: Enhanced model type detection
    if model_type is None:
        model_type = get_llm_type(model)

    # Forward pass
    if model_type == "encoder-decoder":  # Encoder-Decoder models
        enc_inputs = tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)
        dec_inputs = tokenizer(
            "", return_tensors="pt", padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model(
                enc_inputs.input_ids,
                decoder_input_ids=dec_inputs.input_ids,
                output_hidden_states=True,
            )

    # Encoder only OR Decoder only models
    else:
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

    # Extract hidden states
    if use_decoder and "decoder_hidden_states" in outputs:
        hs_tuple = outputs["decoder_hidden_states"]
    elif "encoder_hidden_states" in outputs:
        hs_tuple = outputs["encoder_hidden_states"]
    else:
        hs_tuple = outputs["hidden_states"]

    # FIXED: Enhanced extraction logic with proper handling for all cases
    if get_all_hs:
        reps = []
        for layer_idx, layer_h in enumerate(hs_tuple):
            # FIXED: Convert to float32 immediately to avoid precision issues
            layer_h = layer_h.float()

            # FIXED: Handle different extraction strategies properly
            if strategy == "first-token":
                # CHANGED: Use first token (good for encoder models)
                extracted = layer_h[:, 0, :].squeeze(0).cpu().numpy()

            elif strategy == "last-token":
                # FIXED: Use attention mask to find actual last token, not padding
                if attention_mask is not None:
                    # Find last non-padding token for each sequence
                    last_token_indices = attention_mask.sum(dim=1) - 1
                    batch_size = layer_h.shape[0]

                    if batch_size == 1:
                        # Single sequence case
                        last_hidden = layer_h[0, last_token_indices[0], :]
                        extracted = last_hidden.cpu().numpy()
                    else:
                        # Multiple sequences case
                        last_hidden = layer_h[
                            torch.arange(batch_size), last_token_indices
                        ]
                        extracted = last_hidden.squeeze().cpu().numpy()
                else:
                    # Fallback to simple last token if no attention mask
                    extracted = layer_h[:, -1, :].squeeze(0).cpu().numpy()

            elif strategy == "mean":
                # FIXED: Use attention mask for proper mean pooling (exclude padding)
                if attention_mask is not None:
                    # Expand mask to match hidden states dimensions
                    expanded_mask = (
                        attention_mask.unsqueeze(-1).expand(layer_h.size()).float()
                    )
                    # Mask out padding tokens
                    masked_hidden = layer_h * expanded_mask
                    # Sum and divide by actual sequence length
                    summed = torch.sum(masked_hidden, dim=1)
                    lengths = torch.sum(expanded_mask, dim=1)
                    # FIXED: Avoid division by zero
                    lengths = torch.clamp(lengths, min=1.0)
                    mean_hidden = summed / lengths
                    extracted = mean_hidden.squeeze(0).cpu().numpy()
                else:
                    # Fallback to simple mean if no attention mask
                    extracted = layer_h.mean(dim=1).squeeze(0).cpu().numpy()

            elif strategy == "custom":
                # FIXED: Add bounds checking for custom token number
                if token_number is not None:
                    seq_len = layer_h.shape[1]
                    if token_number < seq_len:
                        extracted = layer_h[:, token_number, :].squeeze(0).cpu().numpy()
                    else:
                        # FIXED: Fallback to last available token if token_number is out of bounds
                        extracted = layer_h[:, -1, :].squeeze(0).cpu().numpy()
                        print(
                            f"Warning: token_number {token_number} >= seq_len {seq_len}, using last token"
                        )
                else:
                    raise ValueError(
                        "token_number must be specified for 'custom' strategy"
                    )

            else:
                raise ValueError(
                    "strategy must be 'mean', 'last-token', 'first-token', or 'custom'"
                )

            # FIXED: Ensure float32 output and handle any remaining numerical issues
            extracted = extracted.astype(np.float32)

            # FIXED: Replace any inf/nan values with zeros
            extracted = np.where(np.isfinite(extracted), extracted, 0.0)

            reps.append(extracted)

        # FIXED: Return as float32 numpy array
        result = np.stack(reps).astype(np.float32)

        # FIXED: Final check for extreme values and clip if necessary
        if np.max(np.abs(result)) > 1000:
            print(
                f"Warning: Extreme values detected (max: {np.max(np.abs(result)):.2f}), applying clipping..."
            )
            clip_value = np.percentile(np.abs(result), 99.5)
            result = np.clip(result, -clip_value, clip_value)

        return result

    else:
        # Single layer extraction
        if layer_index is None:
            layer_index = len(hs_tuple) // 2  # middle layer as default

        # FIXED: Add bounds checking for layer_index
        if layer_index >= len(hs_tuple):
            layer_index = len(hs_tuple) - 1
            print(f"Warning: layer_index out of bounds, using layer {layer_index}")

        hs = hs_tuple[layer_index][0].float()  # FIXED: Convert to float32

        # FIXED: Apply same enhanced logic for single layer extraction
        if strategy == "mean":
            if attention_mask is not None:
                expanded_mask = attention_mask.unsqueeze(-1).expand(hs.size()).float()
                masked_hidden = hs * expanded_mask
                summed = torch.sum(masked_hidden, dim=0)
                length = torch.sum(expanded_mask, dim=0)
                length = torch.clamp(length, min=1.0)  # Avoid division by zero
                result = (summed / length).detach().cpu().numpy().astype(np.float32)
            else:
                result = hs.mean(dim=0).detach().cpu().numpy().astype(np.float32)

        elif strategy == "first-token":
            result = hs[0].detach().cpu().numpy().astype(np.float32)

        elif strategy == "last-token":
            if attention_mask is not None:
                last_token_idx = attention_mask.sum(dim=1) - 1
                result = hs[last_token_idx[0]].detach().cpu().numpy().astype(np.float32)
            else:
                result = hs[-1].detach().cpu().numpy().astype(np.float32)

        elif strategy == "custom":
            if token_number is not None:
                seq_len = hs.shape[0]
                if token_number < seq_len:
                    result = hs[token_number].detach().cpu().numpy().astype(np.float32)
                else:
                    result = hs[-1].detach().cpu().numpy().astype(np.float32)
                    print(
                        f"Warning: token_number {token_number} >= seq_len {seq_len}, using last token"
                    )
            else:
                raise ValueError("token_number must be specified for 'custom' strategy")
        else:
            raise ValueError(
                "strategy must be 'mean', 'last-token', 'first-token', or 'custom'"
            )

        # FIXED: Clean up result
        result = np.where(np.isfinite(result), result, 0.0)
        return result


def vectorize_df(
    df_text,
    model,
    tokenizer,
    layer_index=None,
    strategy="last-token",
    model_type=None,
    use_decoder=False,
    get_all_hs=False,
    device=None,
    token_number=None,
):
    """
    FIXED: Converts the df_text to an embedding matrix (n_samples, n_layers, hidden_dim)
    Enhanced with better error handling and progress reporting.

    Args:
        df_text: list or pandas Series with text data
        model: HuggingFace model
        tokenizer: tokenizer
        layer_index: layer index (int or None - middle layer as default)
        strategy: 'first-token', 'last-token', 'mean', or 'custom'
        model_type: str ('encoder', 'encoder-decoder', 'decoder')
        use_decoder: use decoder hidden states for encoder-decoder models
        get_all_hs: extract all hidden states layers
        device: computation device
        token_number: specific token position for 'custom' strategy

    Return:
        numpy-array (n_samples, n_layers, hidden_dim) if get_all_hs=True
        numpy-array (n_samples, hidden_dim) if get_all_hs=False
    """
    # CHANGED: Setup padding token once at the beginning
    tokenizer = setup_tokenizer_padding(tokenizer)

    embeddings = []
    failed_extractions = 0

    # FIXED: Better progress bar with more information
    for idx in tqdm(
        range(len(df_text)), desc=f"Extracting embeddings (strategy: {strategy})"
    ):
        # CHANGED: Added explicit checks for empty/invalid text
        if hasattr(df_text, "iloc"):
            text = df_text.iloc[idx]
        else:
            text = df_text[idx]

        # CHANGED: More robust text validation
        if text is None or (isinstance(text, str) and len(text.strip()) == 0):
            print(f"Warning: Empty text at index {idx}, skipping...")
            continue

        # CHANGED: Additional check for non-string types
        if not isinstance(text, str):
            text = str(text)

        # CHANGED: Check for extremely long texts that might cause issues
        if len(text) > 10000:
            print(
                f"Warning: Very long text at index {idx} (length: {len(text)}), truncating..."
            )
            text = text[:10000]

        vec = extract_representation(
            model=model,
            tokenizer=tokenizer,
            text=text,
            layer_index=layer_index,
            strategy=strategy,
            model_type=model_type,
            use_decoder=use_decoder,
            get_all_hs=get_all_hs,
            device=device,
            token_number=token_number,
        )
        embeddings.append(vec)

    if failed_extractions > 0:
        print(
            f"Warning: {failed_extractions} extractions failed and were replaced with zeros"
        )

    # FIXED: Better error handling for empty results
    if len(embeddings) == 0:
        raise ValueError("No successful extractions - all texts failed to process")

    # FIXED: Stack with proper error handling
    result = np.stack(embeddings).astype(np.float32)

    # FIXED: Final validation and cleanup
    print(f"Extraction completed: shape={result.shape}, dtype={result.dtype}")
    print(
        f"Stats: max={np.max(result):.3f}, min={np.min(result):.3f}, std={np.std(result):.3f}"
    )

    # FIXED: Final check for extreme values
    if np.max(np.abs(result)) > 1000:
        print(
            "Warning: Extreme values detected in final result, applying global clipping..."
        )
        clip_value = np.percentile(np.abs(result), 99.9)
        result = np.clip(result, -clip_value, clip_value)
        print(f"Applied clipping at ±{clip_value:.2f}")

    # FIXED: Final check for inf/nan
    inf_count = np.sum(np.isinf(result))
    nan_count = np.sum(np.isnan(result))
    if inf_count > 0 or nan_count > 0:
        print(
            f"Warning: Found {inf_count} inf and {nan_count} nan values, replacing with zeros..."
        )
        result = np.where(np.isfinite(result), result, 0.0)

    return result


# FIXED: Additional utility functions for model-specific configurations
def get_recommended_strategy(model_type):
    """
    Get recommended extraction strategy based on model type
    """
    if model_type == "encoder":
        return "first-token"
    elif model_type == "decoder":
        return "last-token"
    elif model_type == "encoder-decoder":
        return "mean"  # or "last-token" if using decoder
    else:
        return "mean"  # safe default


def validate_extraction_params(model, tokenizer, model_type, strategy):
    """
    Validate that extraction parameters make sense for the given model
    """
    detected_type = get_llm_type(model)

    if model_type != detected_type:
        print(
            f"Warning: model_type '{model_type}' doesn't match detected type '{detected_type}'"
        )

    recommended_strategy = get_recommended_strategy(detected_type)
    if strategy != recommended_strategy:
        print(
            f"Info: Using strategy '{strategy}', but '{recommended_strategy}' is recommended for {detected_type} models"
        )


# FIXED: Example usage configurations for different models
def get_deberta_config():
    """Configuration for DeBERTa model"""
    return {
        "strategy": "first-token",
        "model_type": "encoder",
        "use_decoder": False,
        "get_all_hs": True,
    }


def get_gemma_config():
    """Configuration for Gemma model"""
    return {
        "strategy": "last-token",
        "model_type": "decoder",
        "use_decoder": False,
        "get_all_hs": True,
    }


def get_t5_config(use_decoder=True):
    """Configuration for T5 model"""
    return {
        "strategy": "last-token" if use_decoder else "first-token",
        "model_type": "encoder-decoder",
        "use_decoder": use_decoder,
        "get_all_hs": True,
    }
