'''
Copyright © 2020 by Xingbo Dong
xingbod@gmail.com
Monash University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


'''


import random
'''
look up table generator
Xingbo Dong@ Monash university
xingbod@gmail.com

'''
def genLUT(q=8,bin_dim=8,isPerm=False):
  LUT = []
  a = list(range(q))
  if isPerm:
    random.shuffle(a)
  for digit in range(q):
    LUT.append([int(d) for d in bin(a[digit])[2:].zfill(bin_dim)])
  # for digit in range(q):
  #   rint = random.randint(0, pow(2,bin_dim)-1)
  #   # rint = digit
  #   LUT.append([int(d) for d in bin(rint)[2:].zfill(bin_dim)])
  return LUT
