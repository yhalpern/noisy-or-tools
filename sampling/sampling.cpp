#include <iostream>
#include <fstream>
#include "sampling.h"
#include <stdlib.h>
#include "mtrand.h"

using namespace std;


Sampler::Sampler(char* dirname, int sample_seed, char* scheduleName, int numSamples){
    verbose=false;
    counter = NULL;
	
    ifstream f;
    char fname[100];
    seed = sample_seed;

    network_name = dirname;
    schedule_name = scheduleName;
    num_samples = numSamples;

    sprintf(fname, "%s/true_params/priors.txt", dirname);
    f.open(fname);
    f >> P;
    priors = new float[P];
    for (int i = 0; i < P; i++)
    {
        f >> priors[i];
    }
    f.close();
    printf("read %d priors\n", P);
    
    sprintf(fname, "%s/true_params/noise.txt", dirname);
    f.open(fname);
    f >> C;
    noise = new float[C];
    for (int i = 0; i < C; i++)
    {
        f >> noise[i];
    }
    f.close();
    printf("read %d noises\n", C);
    
    sprintf(fname, "%s/true_params/weights.txt", dirname);
    f.open(fname);
    edges = new float*[P];
    for (int i = 0; i < P; i++){
        edges[i] = new float[C];
        for (int j = 0; j < C; j++)
        {
            f >> edges[i][j];
        }
    }

    f.close();
    printf("read weights\n");
}

void Sampler::getSchedule()
{
    char* dirname = network_name;
    char fname[100];
    ifstream f;
    
    sprintf(fname, "%s/true_params/%s", dirname, schedule_name);
    f.open(fname);
    f >> N;

    schedule = new Quartet[N];

    for(int i = 0; i < N; i++)
    {
        f >> schedule[i].a;
        f >> schedule[i].b;
        f >> schedule[i].c;
        f >> schedule[i].d;
    }
    f.close();


    printf("initialized sampler\n");
    printf("P=%d, C=%d, N=%d\n", P, C, N);
    if(verbose)
    {
        printf("initialized sampler\n");
        printf("P=%d, C=%d\n", P, C);
        printf("priors:");
        for(int i = 0; i < P; i++) printf(" %f", priors[i]);
        printf("\n");

        printf("noise:");
        for(int i = 0; i < C; i++) printf(" %f", noise[i]);
        printf("\n");

        printf("weights:");
        for(int i = 0; i < P; i++){
            for(int j = 0; j < C; j++){
                printf(" %f", edges[i][j]);
            }
            printf("\n");
        }
        printf("schedule:");
        for(int i =0; i<N; i++){
            printf("%d %d %d\n", schedule[i].a, schedule[i].b, schedule[i].c);
        }
    }
}

void Sampler::initialize_counter()
{
    if(counter) delete counter;
    counter = new int[16*N];
    for(int i = 0; i < 16*N; i++) counter[i] = 0;
    cout << "counter initialized." << endl;
    return;
}

void Sampler::sample(bool printSamples)
{
    MTRand_closed rnd;
	rnd.seed((unsigned long int)seed);
    char* dirname = network_name;
    int obs[C];
    float r;

    ofstream out;
    char fname[256];
    sprintf(fname, "%s/progress_log_n%d_s%d_%s", network_name, num_samples, seed, schedule_name);
    out.open(fname);

    for(int sample_counter = 0; sample_counter < num_samples; sample_counter++)
    {
        if((sample_counter  % 1000) ==0)
        {
            printf("sample %d\n", sample_counter);
        }
        
        for (int c = 0; c < C; c++){
        //initialize with noise
            r = rnd();
            if(r > noise[c]) obs[c] = 1;
            else obs[c] = 0;
        }

        for(int p = 0; p < P; p++)
        {
            r = rnd();
            if(r < priors[p])
            {
                for(int c = 0; c < C; c++)
                {
                    if(edges[p][c] < 1 && rnd() > edges[p][c]) 
                        obs[c] = 1;
                }
            }
        }
        
        //obs is filled with 0's and 1s now.
        if(printSamples)
        {
            ofstream out;
            char fname[256];
            sprintf(fname, "%s/samples/raw_samples_n%d_s%d", network_name, num_samples, seed);
            if (sample_counter) out.open(fname, ofstream::app);
            else out.open(fname);
            for (int c = 0; c < C-1; c++){
                out << obs[c] << " ";
            }
            out << obs[C-1];
            out << endl;
            out.close();
        }
        
        //time to update counter

        int a,b,c,d;
        for(int s = 0; s < N; s++)
        {
            a = (schedule[s].a >= 0? obs[schedule[s].a]:0);
            b = (schedule[s].b >= 0? obs[schedule[s].b]:0);
            c = (schedule[s].c >= 0? obs[schedule[s].c]:0);
            d = (schedule[s].d >= 0? obs[schedule[s].d]:0);
            counter[16*s + 8*a + 4*b + 2*c+d] += 1;
        }
    }
    out.close();
}


void Sampler::printSamples()
{
    ofstream out;
    char fname[256];
    sprintf(fname, "%s/samples/sufficientStatistics_n%d_s%d_%s", network_name, num_samples, seed, schedule_name);
    cout << "printing sufficient statistics to" << fname << endl;

    out.open(fname);
    out << C << endl;

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

