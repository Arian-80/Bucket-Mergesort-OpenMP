#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

struct Bucket {
    float value;
    struct Bucket* next;
    int count;
};

/*
 * The majority of the sequential code in this project is replicated across..-
 * -.. all versions of this algorithm - MPI, Pthreads and OpenMP.
 */
void initialiseBuckets(struct Bucket* buckets, int bucketCount);
int fillBuckets(const float* floatArrayToSort, int size,
                struct Bucket* buckets, int bucketCount);
void freeBuckets(struct Bucket* buckets, int bucketCount);
void mergesort_parallel(float* array, int size, int threads);
void mergesort(float* array, int low, int high);
void merge(float* floatArrayToSort, int low, int mid, int high);


int bucketsort(float* floatArrayToSort, int arraySize, int threadCount,
               int bucketCount, int threadsPerThread) {
    /*
     * @param floatArrayToSort      Array to sort, self-descriptive
     * @param arraySize             Size of the array to sort
     * @param bucketCount           Number of buckets
     * @param threadsPerThread      Number of threads per thread to perform ..-
     *                              -.. parallel mergesort. 1 = sequential.
     */
    if (bucketCount < 1 || threadCount < 1 || bucketCount > arraySize) {
        printf("Invalid input.\n");
        return 0;
    }
    else if (bucketCount == 1) {
        mergesort(floatArrayToSort, 0, arraySize);
        return 1;
    }

    if (threadCount > bucketCount) threadCount = bucketCount;

    struct Bucket* buckets = (struct Bucket*) malloc((size_t) bucketCount * sizeof(struct Bucket));
    if (!buckets) {
        printf("An error has occurred.\n");
        return 0;
    }

    initialiseBuckets(buckets, bucketCount);
    // negative numbers in list or failure to malloc
    if (!fillBuckets(floatArrayToSort, arraySize, buckets, bucketCount)) {
        printf("An error has occurred.\n");
        return 0;
    }

    float** numbersInBuckets;
    numbersInBuckets = (float**) malloc(bucketCount * sizeof(float*));
    if (!numbersInBuckets) {
        printf("An error has occurred.\n");
        freeBuckets(buckets, bucketCount);
        return 0;
    }

    int sizes[bucketCount];


    omp_set_nested(1);
    omp_set_dynamic(0);

    int itemsInBucket, errorOccurred;
    errorOccurred = 0;
    #pragma omp parallel for \
    private(itemsInBucket) \
    shared(buckets, bucketCount, numbersInBuckets, sizes) \
    shared(errorOccurred, threadsPerThread) \
    default(none) num_threads(threadCount)
    for (int i = 0; i < bucketCount; i++) {
        if (errorOccurred) continue; // Do not proceed anymore if error occurred.
        struct Bucket *currentBucket, *prevBucket;
        currentBucket = &buckets[i];
        itemsInBucket = currentBucket->count;
        sizes[i] = itemsInBucket;
        numbersInBuckets[i] = (float *) malloc(itemsInBucket * sizeof(float));
        if (!numbersInBuckets[i]) {
            for (int j = 0; j < i; j++) free(numbersInBuckets[j]);
            errorOccurred = 1;
            continue;
        }
        if (!itemsInBucket) {
            continue;
        }
        for (int j = 0; j < itemsInBucket; j++) {
            numbersInBuckets[i][j] = currentBucket->value;
            prevBucket = currentBucket;
            currentBucket = currentBucket->next;
            if (j == 0) continue; // first bucket allocated on stack
            // Freeing buckets here saves the need to use another loop to do so.
            free(prevBucket);
        }
        mergesort_parallel(numbersInBuckets[i], itemsInBucket, threadsPerThread);
//            mergesort(numbersInBuckets[i], 0, itemsInBucket - 1);
    }
    free(buckets);
    if (errorOccurred) {
        printf("An error has occurred.\n");
        return -1;
    }
    int k = 0;
    for (int i = 0; i < bucketCount; i++) {
        for (int j = 0; j < sizes[i]; j++) {
            floatArrayToSort[k] = numbersInBuckets[i][j];
            k++;
        }
        free(numbersInBuckets[i]);
    }
    free(numbersInBuckets);
    return 1;
}

void initialiseBuckets(struct Bucket* buckets, int bucketCount) {
    for (int i = 0; i < bucketCount; i++) {
        buckets[i].value = -1;
        buckets[i].next = NULL;
        buckets[i].count = 0;
    }
}

void freeBuckets(struct Bucket* buckets, int bucketCount) {
    struct Bucket *prevBucket, *currentBucket;
    for (int i = 0; i < bucketCount; i++) {
        currentBucket = &buckets[i];
        for (int j = 0; j < currentBucket->count; j++) {
            prevBucket = currentBucket;
            currentBucket = currentBucket->next;
            if (j == 0) continue; // Ignore first bucket allocated on stack
            free(prevBucket);
        }
    }
    free(buckets);
}

int fillBuckets(const float* floatArrayToSort, int size, struct Bucket* buckets, int bucketCount) {
    float currentItem;
    struct Bucket *bucket;
    for (int i = 0; i < size; i++) {
        currentItem = floatArrayToSort[i];
        if (currentItem < 0) {
            freeBuckets(buckets, bucketCount);
            printf("Invalid input: Negative numbers.\n");
            return 0; // No negative numbers allowed
        }
        // ASSUMES NUMBERS WILL BE FROM 0 TO 1. EXPLAIN IN DISS - OVERVIEW OF ALGORITHM. IF NUMBERS ABOVE 1, PERFORMANCE = BAD.
        if (currentItem < 0.9) {
            bucket = &(buckets[(int) (currentItem * 10)]);
        } else { // If larger than limit, store in the final bucket
            bucket = &buckets[bucketCount-1];
        }
        bucket->count++;
        if ((int) bucket->value == -1) {
            bucket->value = currentItem;
            continue;
        }

        struct Bucket *newBucket = (struct Bucket *)
                malloc(sizeof(struct Bucket));
        if (newBucket == NULL) {
            freeBuckets(buckets, bucketCount);
            return 0;
        }
        newBucket->next = bucket->next;
        bucket->next = newBucket;

        newBucket->value = currentItem;
        newBucket->count = 1;
    }
    return 1;
}

void mergesort_parallel(float* array, int size, int threads) {
    if (threads < 2) {
        mergesort(array, 0, size-1);
        return;
    }

    //  EXPLAIN IN DISS. GETTING RID OF TASKS SO THE IMPLEMENTATIONS ARE SIMILAR, ELMINIATING AN EXTERNAL FACTOR THAT COULD PLAY A ROLE IN THE RESULTS.
    omp_set_dynamic(0);
    omp_set_nested(1);
    int portion = size / threads;
    int remainder = size % threads;
    int starts[threads];
    int portions[threads];
    #pragma omp parallel num_threads(threads)
    {
        int start, end, rank;
        rank = omp_get_thread_num();
        if (rank < remainder) {
            start = rank * (portion + 1);
            end = start + portion + 1;
            portions[rank] = portion + 1;
        } else {
            start = portion * (rank - remainder) + remainder * (portion + 1);
            end = start + portion;
            portions[rank] = portion;
        }
        starts[rank] = start;
        mergesort(array, start, end-1);
    }
    /* Final merges */
    int low, mid, high, temp;
    low = 0;
    high = portions[0]-1;
    for (int i = 0; i < threads-1; i++) {
        temp = portions[i+1]-1 + starts[i+1];
        mid = high;
        high = temp;
        merge(array, low, mid, high);
    }
}

void mergesort(float* array, int low, int high) {
    if (low >= high) return;
    int mid = low + (high - low)/2;
    mergesort(array, low, mid); // low -> mid inclusive
    mergesort(array, mid + 1, high);
    merge(array, low, mid, high);
}

void merge(float* floatArrayToSort, int low, int mid, int high) {
    int i, j, k;
    int lengthOfA = mid - low + 1; // low -> mid, inclusive
    int lengthOfB = high - mid;
    float *a, *b;
    a = malloc(lengthOfA * sizeof(float));
    if (a) {
        b = malloc(lengthOfB * sizeof(float));
        if (!b) {
            free(a);
            return;
        }
    }
    else {
        return;
    }

    for (i = 0; i < lengthOfA; i++) {
        a[i] = floatArrayToSort[i + low];
    }
    for (j = 0; j < lengthOfB; j++) {
        b[j] = floatArrayToSort[j + mid + 1];
    }
    i = j = 0;
    k = low;
    while (i < lengthOfA && j < lengthOfB) {
        if (a[i] <= b[j]) {
            floatArrayToSort[k] = a[i];
            i++;
        }
        else {
            floatArrayToSort[k] = b[j];
            j++;
        }
        k++;
    }
    for (;i < lengthOfA; i++) {
        floatArrayToSort[k] = a[i];
        k++;
    }
    for (;j < lengthOfB; j++) {
        floatArrayToSort[k] = b[j];
        k++;
    }
}

int main() {
    for (int k = 1; k < 9; k++) {
        if (k == 3) k = 4;
        if (k == 5) k = 6;
        if (k == 7) k = 8;
        int size = 1000000;
        float *array = (float *) malloc((size_t) size * sizeof(float));
        if (array == NULL) return -1;
        time_t t;
        srand((unsigned) time(&t));
        for (int i = 0; i < size; i++) {
            array[i] = (float) rand() / (float) RAND_MAX;
        }
        int incorrectCounter, correctCounter;
        incorrectCounter = correctCounter = 0;
        for (int i = 1; i < size; i++) {
            if (array[i] < array[i - 1]) incorrectCounter++;
            else correctCounter++;
        }
        correctCounter++; // final unaccounted number
//        printf("Initially sorted numbers: %d\nIncorrectly sorted numbers: %d\nTotal numbers: %d\n",
//               correctCounter, incorrectCounter, size);
        double start, end;
        int result;
        start = omp_get_wtime();
        result = bucketsort(array, size, 1, 1, k);
        end = omp_get_wtime();
        if (!result) {
            free(array);
            return 0;
        }
        incorrectCounter = correctCounter = 0;
        for (int i = 1; i < size; i++) {
            if (array[i] < array[i - 1]) incorrectCounter++;
            else correctCounter++;
        }
        correctCounter++; // final unaccounted number
        printf("Sorted numbers: %d\nIncorrectly sorted numbers: %d\nTotal numbers: %d\n",
               correctCounter, incorrectCounter, size);
        printf("Time taken: %g\n", end - start);
        FILE *f = fopen("times.txt", "a");
        fprintf(f, "%g,", end - start);
        free(array);
    }
    return 0;
}