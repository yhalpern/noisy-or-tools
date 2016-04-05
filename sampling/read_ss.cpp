#include "statistics.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>

using namespace std;
int main(int argc, char* argv[])
{
    if(argc < 8)
    {
        cout << "usage: ./read_ss network raw_samples schedule n C offset seed" << endl;
        exit(1);
    }

    char * network_name = argv[1];
    char * raw_samples_name = argv[2];
    char * schedule_name = argv[3];
    int n = atoi(argv[4]);
    int C = atoi(argv[5]);
    int offset = atoi(argv[6]);
    int seed = atoi(argv[7]);

    cout << "reading ss from " << raw_samples_name << " file with " <<n << " samples" << "starting at position" <<offset <<endl;
    Stats s = Stats(network_name, seed, schedule_name, n, C, raw_samples_name);
    cout << "initialized" << endl;
    s.getSchedule();
    cout << "read schedule" << endl;
    s.initialize_counter();
    cout << "initialized" << endl;

    s.read(offset);
    cout << "read data" << endl;
    s.printStats(); 
}
