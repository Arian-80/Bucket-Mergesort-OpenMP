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
void mergesort(float* array, int low, int high);
void merge(float* floatArrayToSort, int low, int mid, int high);


int bucketsort(float* floatArrayToSort, int arraySize, int threadCount,
               int bucketCount) {
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
    shared(errorOccurred) \
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
        }
        else {
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
            mergesort(numbersInBuckets[i], 0, itemsInBucket - 1);
        }
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
    float bucketLimit = 0.1 * bucketCount;
    for (int i = 0; i < size; i++) {
        currentItem = floatArrayToSort[i];
        if (currentItem < 0) {
            freeBuckets(buckets, bucketCount);
            printf("Invalid input: Negative numbers.\n");
            return 0; // No negative numbers allowed
        }
        if (currentItem < bucketLimit) {
            bucket = &(buckets[(int) (currentItem * 10)]);
        } else { // If larger than limit, store in the final bucket
            bucket = &buckets[bucketCount-1];
        }
        bucket->count++;
        if ((int) bucket->value == -1) {
            bucket->value = currentItem;
            continue;
        }
        while (bucket->next != NULL) {
            bucket = bucket->next;
        }
        struct Bucket *newBucket = (struct Bucket *)
                malloc(sizeof(struct Bucket));
        if (newBucket == NULL) {
            freeBuckets(buckets, bucketCount);
            return 0;
        }
        bucket->next = newBucket;
        newBucket->value = currentItem;
        newBucket->next = NULL;
        newBucket->count = 0;
    }
    return 1;
}

void mergesort(float* array, int low, int high) {
    if (low >= high) return;
    int mid = low + (high - low)/2;
    #pragma omp task default(none) \
    shared(array, low, mid)
    {
        mergesort(array, low, mid); // low -> mid inclusive
    }
    #pragma omp task default(none) \
    shared(array, mid, high)
    {
        mergesort(array, mid + 1, high);
    }
    #pragma omp taskwait
    merge(array, low, mid, high);
}

void merge(float* floatArrayToSort, int low, int mid, int high) {
    int i, j, k;
    int lengthOfA = mid - low + 1; // low -> mid, inclusive
    int lengthOfB = high - mid;
    float a[lengthOfA], b[lengthOfB];
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
    int size = 10000;
    float* array = (float*) malloc((size_t) size * sizeof(float));
    if (array == NULL) return -1;
    time_t t;
    srand((unsigned) time(&t));
    for (int i = 0; i < size; i++) {
        array[i] = (float) rand() / (float) RAND_MAX;
    }
    int incorrectCounter, correctCounter;
    incorrectCounter = correctCounter = 0;
    for (int i = 1; i < size; i++) {
        if (array[i] < array[i-1]) incorrectCounter++;
        else correctCounter++;
    }
    correctCounter++; // final unaccounted number
//    printf("Initially sorted numbers: %d\nIncorrectly sorted numbers: %d\nTotal numbers: %d\n",
//           correctCounter, incorrectCounter, size);
    double start, end;
    start = omp_get_wtime();
    if (!bucketsort(array, size, 8, 10)) {
        free(array);
        return 0;
    }
    end = omp_get_wtime();
    incorrectCounter = correctCounter = 0;
    for (int i = 1; i < size; i++) {
        if (array[i] < array[i-1]) incorrectCounter++;
        else correctCounter++;
    }
    correctCounter++; // final unaccounted number
    printf("Sorted numbers: %d\nIncorrectly sorted numbers: %d\nTotal numbers: %d\n",
           correctCounter, incorrectCounter, size);
    printf("Time taken: %g\n", end-start);
    free(array);
    return 0;
}