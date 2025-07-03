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

# Inspiration

The boundry for checking the conjecture does not seem to have been pushed since this paper in 2014
[T. OLIVEIRA e SILVA, S. HERZOG, AND S. PARDI](https://www.ams.org/journals/mcom/2014-83-288/S0025-5718-2013-02787-1/S0025-5718-2013-02787-1.pdf)

Their code is not available, so this is my attempt at getting close on a laptop. 

With the speed previously described it will take around 700 GPU years from a 3070 ti to get to 10e17. 

This is obviously not as fast as the code in the paper as they only used around that amount of CPU years in 2014 to check up to 4 * 10e17. 

Contributions are welcome in the from of pull requests, but please discuss them in an issue before spending time implementing. 
