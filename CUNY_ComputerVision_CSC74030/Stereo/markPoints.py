import stereo

# create control points
p = stereo.Pairs(lfile='ctrl_pl.npy', rfile='ctrl_pr.npy', nPnt=40)
p.captureCoor()
print("Well Done!")

# create test points
# p = stereo.Pairs(lfile='test_pl.npy', rfile='test_pr.npy', nPnt=8)
# p.captureCoor()
# print("Well Done!")