#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include "statistics.h"
#include <stdlib.h>
#include "mtrand.h"

using namespace std;


Stats::Stats(char* dirname, int sample_seed, char* scheduleName, int numSamples, int C, char* raw_samples){
    verbose=false;
    counter = NULL;
    printf("successfully initialized counter to null %x\n", counter);
    this->C = C;
    seed = sample_seed;
    network_name = dirname;
    schedule_name = scheduleName;
    raw_samples_name = raw_samples;
    num_samples = numSamples;
}

Stats::~Stats(){
  if(schedule) delete schedule;
  if(counter) delete counter;
}


void Stats::getSchedule()
{
    char* dirname = network_name;
    char fname[500];
    ifstream f;
    
    sprintf(fname, "%s/true_params/%s", dirname, schedule_name);
    f.open(fname);
    f >> N;
    cout << "N is "<<N << endl;
    schedule = new Quartet[N];

    for(int i = 0; i < N; i++)
    {
        f >> schedule[i].a;
        f >> schedule[i].b;
        f >> schedule[i].c;
        f >> schedule[i].d;
    }
    f.close();
    printf("initialized stats reader\n");   
}

void Stats::initialize_counter()
{
    
    if(counter) delete counter;
    counter = new int[16*N];
    for(int i = 0; i < 16*N; i++) counter[i] = 0;
    return;
}

void Stats::read(int offset)
{
    bool reading = true;
    MTRand rnd;
	rnd.seed((unsigned long int)seed);
    printf("reading\n");
    char* dirname = network_name;
    int obs[C];
    for (int i=0; i < C; i++) obs[i] = 0;
    float r;
    int a,b,c,d;

    ofstream out;
    char fname[256];
    string line;
    stringstream ss;   
    vector<string> tokens;
    sprintf(fname, "%s/progress_log_n%d_s%d_%s", network_name, num_samples, seed, schedule_name);
    out.open(fname);

    ifstream in;
    if(reading)
    {
        printf("reading %d samples from %s, starting at %d and ending at %d\n", num_samples, raw_samples_name, offset, offset+num_samples);
        in.open(raw_samples_name);
    }

    for(int sample_counter = 0; sample_counter < offset+num_samples; sample_counter++)
    {
        if((sample_counter  % 1000) ==0 && sample_counter > offset)
        {
            printf("sample %d\n", sample_counter);
        }


        
        getline(in, line);
        //printf("read line %s\n", line.c_str());
        
        tokens.clear();
        ss.clear();
        ss.str(line);
        char buf[256];

        while (ss >> buf)
            tokens.push_back(buf);

        if (string("compact").compare(tokens[1]) == 0)
        {
            //printf("compact: \"%s\"\n", tokens[1].c_str());
            for (int i=2; i< tokens.size(); i++)
            {
                //printf("setting obs[%s] to 1\n", tokens[i].c_str());
                obs[atoi(tokens[i].c_str())] = 1;
            }

        }
        else
        {
            //printf("original: \"%s\":\"compact\"\n", tokens[1].c_str(), "compact");
            //for (int i=0; i < tokens[1].length(); i++)
            //{
            //    printf("%c:%c\n", "compact"[i], tokens[1].c_str()[i]);
            //}
            for(int i=0; i<C; i++)
            {
                obs[i] = atoi(tokens[i].c_str());
            }
        }
    
        //time to update counter
        if (sample_counter >= offset)
        {
            for(int s = 0; s < N; s++)
            {
                a = (schedule[s].a >= 0? obs[schedule[s].a]:0);
                b = (schedule[s].b >= 0? obs[schedule[s].b]:0);
                c = (schedule[s].c >= 0? obs[schedule[s].c]:0);
                d = (schedule[s].d >= 0? obs[schedule[s].d]:0);
                counter[16*s + 8*a + 4*b + 2*c+d] += 1;
            }
        }
        
        //cleanup
        if ("compact" == tokens[1])
        {
            for (int i=2; i< tokens.size(); i++)
            {
                obs[atoi(tokens[i].c_str())] = 0;
            }
        }
        //printf("done\n");
    }
    out.close();
}


void Stats::printStats()
{
    ofstream out;
    char fname[256];
    sprintf(fname, "%s/samples/sufficientStatistics_n%d_s%d_%s", network_name, num_samples, seed, schedule_name);
    cout << "printing sufficient statistics to " << fname << endl;

    out.open(fname);
    out << C << endl;
    cout << "here" << endl;
    for(int i = 0; i < 16*N; i++){
        if(i>0 && i%16==0) out << endl;
        out << counter[i] << " ";
    }
    out.close();
    
    sprintf(fname, "%s/samples/id_n%d_s%d_%s", network_name, num_samples, seed, schedule_name);
    out.open(fname);
    for(int i = 0; i < N; i++){
        out << schedule[i].a << " "<< schedule[i].b << " " << schedule[i].c << " " << schedule[i].d << endl;;
    }
    out.close();
}

