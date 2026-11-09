"""Loss-boundary regression: supervise label and EOS, never prompt or padding."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import mlx.core as mx
from train_lora import completion_mask,completion_loss


class CompletionLossTests(unittest.TestCase):
    def test_exact_mask_and_padding_invariance(self):
        batch=mx.array([[1,2,3,4,5,0,0]])
        lengths=mx.array([[2,5]])
        self.assertEqual(completion_mask(batch,lengths).tolist(),[[False,True,True,True,False,False]])
        class Model:
            def __init__(self,poison): self.poison=poison
            def __call__(self,tokens):
                logits=mx.zeros((1,6,8))
                # Alter only ignored positions, including first post-EOS pad.
                logits[:,mx.array([0,4,5]),:]=self.poison*mx.arange(8)
                return logits
        loss1,n=completion_loss(Model(0),batch,lengths)
        loss2,_=completion_loss(Model(100),batch,lengths)
        self.assertEqual(n.item(),3)
        self.assertAlmostEqual(loss1.item(),loss2.item(),places=6)


if __name__=='__main__': unittest.main()
