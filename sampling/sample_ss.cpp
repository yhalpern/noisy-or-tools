#include "sampling.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>

using namespace std;
int main(int argc, char* argv[])
{
    if(argc < 5)
    {
        cout << "usage: ./sample_ss network schedule seed n" << endl;
        exit(1);
    }

    char * network_name = argv[1];
    char * schedule_name = argv[2];
    int seed = atof(argv[3]);
    int n = atoi(argv[4]);

    cout << "sampling ss from " << network_name << " network taking " << n << " samples with seed " << seed << endl;

    Sampler s = Sampler(network_name, seed, schedule_name, n);
    
    s.getSchedule();
    s.initialize_counter();
    s.sample(false);
    s.printSamples(); 
}
