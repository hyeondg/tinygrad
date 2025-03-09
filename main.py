from tinygrad import Tensor
import time
N = 8192

flop = N*N*2*N
print(f"{flop / 1e9:.2f} GFLOP")

for t in range(1):
    print(">>>> HERE 1")
    t1 = Tensor.rand(N, N).realize()
    print(">>>> HERE 2")
    t2 = Tensor.rand(N, N).realize()
    print(">>>> HERE 3")

    st = time.monotonic()
    print(">>>> HERE 4")
    t3 = t1.dot(t2)
    print(">>>> HERE 5")
    t3.realize()
    print(">>>> HERE 6")
    et = time.monotonic()
    print(">>>> HERE 7")
    s = (et-st)
    print(">>>> HERE 8")
    print(f"{flop/s * 1e-9:.2f} GLOPS")
    print(">>>> HERE 9")