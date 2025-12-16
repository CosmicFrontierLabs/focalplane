/*
 * Reference implementation test harness for CF_LOS_FB_40Hz
 *
 * This program runs the C implementation of the LOS feedback controller
 * with known inputs and outputs the results for comparison with the Rust
 * implementation.
 *
 * Compile with:
 *   gcc -O2 -o los_test main.c CF_LOS_FB_40Hz.c -lm
 *
 * Run with:
 *   ./los_test > reference_output.csv
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "CF_LOS_FB_40Hz.h"

/*
 * Reset the controller by calling it with PC_onoff_flag = 0
 * (This clears the static variables inside the function)
 */
void reset_controller(void)
{
    double dummy_x, dummy_y;
    CF_LOS_FB_function(&dummy_x, &dummy_y, 0, 0, 0.0, 0.0, 0.0, 0.0);
}

/*
 * Get current time in nanoseconds
 */
long long get_time_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

int main(void)
{
    double U_X, U_Y;
    int i;
    long long start_time, end_time, total_time;
    int num_iterations = 1000;

    /* Print header */
    fprintf(stderr, "LOS Feedback Controller - C Reference Implementation\n");
    fprintf(stderr, "====================================================\n\n");

    /*
     * Test 1: Step response
     * Command = (0, 0), Measurement starts at (1, 1) - constant error of -1
     */
    fprintf(stderr, "Test 1: Step response (100 iterations, constant error)\n");
    printf("# Test 1: Step response\n");
    printf("# step,meas_x,meas_y,cmd_x,cmd_y,u_x,u_y\n");

    reset_controller();

    for (i = 0; i < 100; i++) {
        CF_LOS_FB_function(&U_X, &U_Y,
                           1,          /* enabled */
                           0,          /* not held */
                           0.0, 0.0,   /* command at origin */
                           1.0, 1.0);  /* measurement offset by 1 pixel */
        printf("%d,1.0,1.0,0.0,0.0,%.15e,%.15e\n", i, U_X, U_Y);
    }

    /*
     * Test 2: Varying input (sinusoidal disturbance)
     */
    fprintf(stderr, "Test 2: Sinusoidal disturbance (200 iterations)\n");
    printf("\n# Test 2: Sinusoidal disturbance\n");
    printf("# step,meas_x,meas_y,cmd_x,cmd_y,u_x,u_y\n");

    reset_controller();

    for (i = 0; i < 200; i++) {
        double t = i / 40.0; /* time in seconds at 40Hz */
        double meas_x = 2.0 * sin(2.0 * 3.14159265358979 * 1.0 * t); /* 1 Hz sine */
        double meas_y = 1.5 * cos(2.0 * 3.14159265358979 * 0.5 * t); /* 0.5 Hz cosine */

        CF_LOS_FB_function(&U_X, &U_Y,
                           1,          /* enabled */
                           0,          /* not held */
                           0.0, 0.0,   /* command at origin */
                           meas_x, meas_y);
        printf("%d,%.15e,%.15e,0.0,0.0,%.15e,%.15e\n", i, meas_x, meas_y, U_X, U_Y);
    }

    /*
     * Test 3: Hold mode test
     */
    fprintf(stderr, "Test 3: Hold mode (20 iterations)\n");
    printf("\n# Test 3: Hold mode\n");
    printf("# step,hold_flag,meas_x,u_x,u_y\n");

    reset_controller();

    /* Run for 5 steps to build up state */
    for (i = 0; i < 5; i++) {
        CF_LOS_FB_function(&U_X, &U_Y, 1, 0, 0.0, 0.0, 1.0, 1.0);
        printf("%d,0,1.0,%.15e,%.15e\n", i, U_X, U_Y);
    }

    /* Enable hold for 5 steps */
    for (i = 5; i < 10; i++) {
        CF_LOS_FB_function(&U_X, &U_Y, 1, 1, 0.0, 0.0, 5.0, 5.0); /* different meas, should be ignored */
        printf("%d,1,5.0,%.15e,%.15e\n", i, U_X, U_Y);
    }

    /* Release hold */
    for (i = 10; i < 20; i++) {
        CF_LOS_FB_function(&U_X, &U_Y, 1, 0, 0.0, 0.0, 1.0, 1.0);
        printf("%d,0,1.0,%.15e,%.15e\n", i, U_X, U_Y);
    }

    /*
     * Test 4: Performance benchmark
     */
    fprintf(stderr, "\nTest 4: Performance benchmark (%d iterations)\n", num_iterations);

    reset_controller();

    /* Warm up */
    for (i = 0; i < 100; i++) {
        CF_LOS_FB_function(&U_X, &U_Y, 1, 0, 0.0, 0.0, 1.0, 1.0);
    }

    reset_controller();

    /* Timed run */
    start_time = get_time_ns();
    for (i = 0; i < num_iterations; i++) {
        CF_LOS_FB_function(&U_X, &U_Y, 1, 0, 0.0, 0.0, 1.0, 1.0);
    }
    end_time = get_time_ns();

    total_time = end_time - start_time;
    fprintf(stderr, "  Total time: %lld ns\n", total_time);
    fprintf(stderr, "  Per iteration: %.2f ns\n", (double)total_time / num_iterations);
    fprintf(stderr, "  Iterations per second: %.0f\n",
            1e9 / ((double)total_time / num_iterations));

    printf("\n# Performance: %lld ns total, %.2f ns/iter\n",
           total_time, (double)total_time / num_iterations);

    fprintf(stderr, "\nDone! Output written to stdout (redirect to file for comparison)\n");

    return 0;
}
