using namespace std;

typedef struct Triplet
{
    int a,b,c;
} Triplet;

typedef struct Quartet
{
    int a,b,c,d;
} Quartet;

class Sampler {

    bool verbose;

    int P;
    int C;
    int N;
    int seed;
    float* noise;
    float* priors;
    float** edges;

    char* network_name;
    char* schedule_name;
    char* raw_samples_name;
    int num_samples;

    Quartet* schedule;
    int *counter;

public:
    Sampler(char* dirname, int seed, char* schedule_name, int num_samples);
    void getSchedule();
    void initialize_counter();
    void sample(bool printSamples);
    void printSamples();
};


