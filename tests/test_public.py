import tempfile
import unittest
from pathlib import Path
import torch
from vorafuel import VORAFuel, resolve_device
from unittest.mock import patch

class PublicInterface(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(13)
        self.m=VORAFuel().eval()
        self.h=torch.randn(2,36,14,3)
        self.f=torch.randn(2,1539)
        self.p=torch.randn(2,12,9)

    def test_original_equations_and_checkpoint(self):
        with torch.inference_mode():
            c,r=self.m.policy.backbone.encode(self.h,self.f)
            original=.3*self.m.policy.predict_encoded((c,r,torch.zeros(2,14),torch.zeros(2,1)),self.p)
            actual=self.m(self.h,self.f,self.p)
        torch.testing.assert_close(actual,original,rtol=0,atol=0)
        self.assertTrue(torch.isfinite(actual).all() and (actual>0).all())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.pt';torch.save({'state':self.m.policy.state_dict()},path)
            restored=VORAFuel.from_checkpoint(path)
            with torch.inference_mode():
                torch.testing.assert_close(actual,restored(self.h,self.f,self.p),rtol=0,atol=0)

    def test_prefix_and_batch_independence(self):
        with torch.inference_mode():
            y=self.m(self.h,self.f,self.p)
            p=self.p.clone();p[:,6:]+=10
            changed=self.m(self.h,self.f,p)
            single=self.m(self.h[:1],self.f[:1],self.p[:1])
        torch.testing.assert_close(y[:,:6],changed[:,:6],rtol=0,atol=0)
        torch.testing.assert_close(y[:1],single,rtol=1e-5,atol=1e-6)

    def test_invalid_input_rejected(self):
        self.h[0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):self.m(self.h,self.f,self.p)

    def test_explicit_cuda_fails_without_device(self):
        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(resolve_device('auto').type,'cpu')
            with self.assertRaises(RuntimeError):resolve_device('cuda')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA not available')
    def test_cuda_checkpoint_forward_and_backward(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'model.pt';torch.save({'state':self.m.policy.state_dict()},p)
            gpu=VORAFuel.from_checkpoint(p,device='cuda')
            with torch.inference_mode():cpu=self.m(self.h,self.f,self.p)
            result=gpu(self.h.cuda(),self.f.cuda(),self.p.cuda())
            torch.testing.assert_close(cpu,result.cpu(),rtol=2e-4,atol=2e-5)
            result.mean().backward()
            grads=[p.grad for p in gpu.parameters() if p.requires_grad]
            self.assertTrue(any(g is not None for g in grads))
            self.assertTrue(all(g is None or torch.isfinite(g).all() for g in grads))

if __name__=='__main__':unittest.main()
