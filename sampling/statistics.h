using namespace std;

typedef struct Triplet
{
    int a,b,c;
} Triplet;

typedef struct Quartet
{
    int a,b,c,d;
} Quartet;

class Stats {

    bool verbose;
    bool reading;

    int P;
    int C;
    int seed;
    float* noise;
    float* priors;
    float** edges;

    char* network_name;
    char* schedule_name;
    char* raw_samples_name;
    int num_samples;

    Quartet* schedule;

public:
    int N;
    int *counter;
    Stats(char* dirname, int seed, char* schedule_name, int num_samples, int C, char* raw_samples);
    ~Stats();
    void getSchedule();
    void initialize_counter();
    void read(int offset);
    void printStats();
};


