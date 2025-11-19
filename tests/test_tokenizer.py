"""
测试RNA tokenizer
"""

import torch
import pytest
from rna_fitness.utils.tokenizer import RNATokenizer


def test_tokenizer_init():
    """测试tokenizer初始化"""
    tokenizer = RNATokenizer()
    assert tokenizer.vocab_size == 8
    assert '<PAD>' in tokenizer.vocab
    assert 'A' in tokenizer.vocab
    assert 'U' in tokenizer.vocab
    assert 'G' in tokenizer.vocab
    assert 'C' in tokenizer.vocab


def test_encode_single_sequence():
    """测试编码单个序列"""
    tokenizer = RNATokenizer()
    sequence = "AUGC"
    
    # 不返回tensor
    encoded = tokenizer.encode(sequence, max_length=10, return_tensors=False)
    assert len(encoded) == 1
    assert len(encoded[0]) == 10  # 应该被填充到max_length
    
    # 返回tensor
    encoded_tensor = tokenizer.encode(sequence, max_length=10, return_tensors=True)
    assert isinstance(encoded_tensor, torch.Tensor)
    assert encoded_tensor.shape == (1, 10)


def test_encode_multiple_sequences():
    """测试编码多个序列"""
    tokenizer = RNATokenizer()
    sequences = ["AUGC", "UUUU", "GGCC"]
    
    encoded = tokenizer.encode(sequences, max_length=10, return_tensors=True)
    assert encoded.shape == (3, 10)


def test_encode_decode():
    """测试编码解码循环"""
    tokenizer = RNATokenizer()
    original_sequence = "AUGCAUGC"
    
    # 编码
    encoded = tokenizer.encode(original_sequence, max_length=20, return_tensors=True)
    
    # 解码
    decoded = tokenizer.decode(encoded[0])
    
    # 检查解码后的序列（去除填充）
    assert decoded == original_sequence


def test_dna_to_rna_conversion():
    """测试DNA到RNA的转换（T -> U）"""
    tokenizer = RNATokenizer()
    dna_sequence = "ATGC"  # DNA序列
    
    encoded = tokenizer.encode(dna_sequence, max_length=10, return_tensors=True)
    decoded = tokenizer.decode(encoded[0])
    
    # 应该转换为RNA
    assert 'T' not in decoded
    assert decoded == "AUGC"


def test_batch_encode():
    """测试批量编码"""
    tokenizer = RNATokenizer()
    sequences = ["AUGC", "UUUUAAAA", "GGCCGGCC"]
    
    batch = tokenizer.batch_encode(sequences, max_length=20)
    
    assert 'input_ids' in batch
    assert 'attention_mask' in batch
    assert batch['input_ids'].shape == (3, 20)
    assert batch['attention_mask'].shape == (3, 20)
    
    # 检查attention mask
    # 第一个序列应该有更少的非填充token
    assert batch['attention_mask'][0].sum() < batch['attention_mask'][1].sum()


def test_unknown_token():
    """测试未知token的处理"""
    tokenizer = RNATokenizer()
    sequence = "AUXN"  # X和N是未知碱基
    
    encoded = tokenizer.encode(sequence, max_length=10, return_tensors=False)
    
    # 应该包含UNK token
    assert tokenizer.vocab['<UNK>'] in encoded[0]


def test_max_length_truncation():
    """测试最大长度截断"""
    tokenizer = RNATokenizer()
    long_sequence = "AUGC" * 100  # 很长的序列
    max_length = 20
    
    encoded = tokenizer.encode(long_sequence, max_length=max_length, return_tensors=True)
    
    # 应该被截断到max_length
    assert encoded.shape[1] == max_length


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
