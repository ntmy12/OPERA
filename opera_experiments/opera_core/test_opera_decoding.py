import torch
import unittest
from opera_decoding import OverTrustPenaltyLogitsProcessor, RetrospectionAllocator

class TestOPERADecoding(unittest.TestCase):
    def test_over_trust_penalty(self):
        processor = OverTrustPenaltyLogitsProcessor(scale_factor=2.0, penalty_weights=1.0)
        
        # Mock attention: batch_size=1, num_heads=2, seq_len=4, seq_len=4
        # High attention on the first token (image token simulation)
        mock_attn = torch.zeros(1, 2, 4, 4)
        mock_attn[0, :, 3, 0] = 0.9  # High attention from token 3 to token 0
        processor.update_attention(mock_attn)
        
        input_ids = torch.tensor([[1, 2, 3, 4]])
        scores = torch.ones(1, 10)  # Mock vocab scores
        
        penalized_scores = processor(input_ids, scores)
        
        # Max agg score is 0.9, scale_factor is 2.0 -> penalty = 1.8
        # Original score = 1.0, penalized = 1.0 - 1.8 = -0.8
        self.assertTrue(torch.allclose(penalized_scores, torch.tensor(-0.8).expand(1, 10)))

    def test_retrospection_allocator(self):
        allocator = RetrospectionAllocator(threshold=0.8)
        
        # Below threshold
        mock_attn_normal = torch.zeros(1, 2, 4, 4)
        mock_attn_normal[0, :, 3, 0] = 0.5
        self.assertFalse(allocator.check_rollback(mock_attn_normal))
        
        # Above threshold
        mock_attn_high = torch.zeros(1, 2, 4, 4)
        mock_attn_high[0, :, 3, 0] = 0.9
        self.assertTrue(allocator.check_rollback(mock_attn_high))

if __name__ == "__main__":
    unittest.main()
