"""本地压缩器的统一入口；导入本包不会导入 PyTorch 或加载权重。"""

LOCAL_METHODS = ("selective_context", "llmlingua2", "salt_summary")


def create_compressor(method, device='cpu'):
    """只创建所选方法。返回对象的 compress(context, retained_ratio) 输出字符串。"""
    if method == "selective_context":
        from .selective_context import SelectiveContextCompressor
        return SelectiveContextCompressor(device=device)
    if method == "llmlingua2":
        from .llmlingua2 import LLMLingua2Compressor
        return LLMLingua2Compressor(device=device)
    if method == "salt_summary":
        from .salt_summary import SaltSummaryCompressor
        return SaltSummaryCompressor(device=device)
    raise ValueError(f"Unknown local compressor: {method}")


def validate_input(context, retained_ratio):
    if not isinstance(context, str):
        raise TypeError("context must be a string, not an example containing a question/label.")
    if isinstance(retained_ratio, bool) or not 0 < retained_ratio <= 1:
        raise ValueError("retained_ratio is the fraction to KEEP, in (0, 1].")
