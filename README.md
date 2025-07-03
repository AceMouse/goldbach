# Setup
set up python and nvcc from the nvidia toolkit.
Then run
```
./setup.sh
```
Now you can run the following command to check that all even numbers up to top can be written a sum of primes a and b where a is less than delta. 

Delta is usually fine to set at 10e4. 
```
./check <delta> <top>
```
This command will only print something if there is a failure.

To see this try running with delta at 10 and top at 100. 

You should see 98 fail because it needs 98=19+79, no smaller prime than 19 will do, it will also print the primes it tried to use to create the sum. 

On my NVIDIA GeForce RTX 2070 Super Max-Q I can check around 3.5e+7 numbers per sec at peak. On a 3070 ti is around 4.5e+7. This includes the skipped odd numbers. 
