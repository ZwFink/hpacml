import sys
sys.path.append('/usr/workspace/koparasy/approx-programming-model/approx-llvm/approx/')
from approx_modules import approx

approxDataProfile = approx.approxApplication(sys.argv[1])
print(approxDataProfile.getApplicationInput())
print(approxDataProfile.getApplicationOutput())
for r in approxDataProfile:
  print (r)
  X = r.X()
  Y = r.Y()
  print(X)
  print(Y)
