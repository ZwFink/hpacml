import sys
sys.path.append('/usr/workspace/fink12/approx_hpc/approx-llvm/approx/')
from approx_modules import approx

approxDataProfile = approx.approxApplication(sys.argv[1])
print(approxDataProfile.getApplicationInput())
print(approxDataProfile.getApplicationOutput())
for r in approxDataProfile:
  print (r)
  X = r.X()
  Y = r.Y()
  print(X, X.shape)
  print(Y, Y.shape)
