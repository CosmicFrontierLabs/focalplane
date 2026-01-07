/*

NOMINAL OPERATION:
    1.) Start with PC_onoff_flag = 0 and HOLD_flag = 0.
    2.) Set LOS_CMD_X = 0.0 [pixels] and LOS_CMD_Y = 0.0 [pixels] or to the
desired detector regulation point at start up. 3.) Set PC_onoff_flag = 1. This
will activate the servo. 4.) Set HOLD_flag = 1 to pause control if needed. (Hold
asserted does not reset filter states.) Reset to 0 to resume control. 5.) Set
LOS_CMD_X [pixels] and LOS_CMD_Y [pixels] to nonzero values to jog the centroid
around the detector focal plane. 6.) Shutdown Procedure: Set PC_onoff_flag = 0.
This also resets the filter states to zero.

INPUTS:
    PC_onoff_flag: 0 or 1 int value. Used to activate (1) or deactivate (0) the
FB controller. HOLD_flag: 0 or 1 int value. Used to pause the controller (1).
               Set back to 0 to pick up where you left off.
    LOS_CMD_X: Units (pixels). Double data type. Start as 0.0. X servo command.
    LOS_CMD_Y: Units (pixels). Double data type. Start as 0.0. Y servo command.
    LOS_MEAS_X: Units (pixels). Double data type. X centroid measurement.
    LOS_MEAS_Y: Units (pixels). Double data type. Y centroid measurement.

OUTPUTS:
    *U_X: Units (urad). Pointer to a double. Command to FSM electronics.
    *U_Y: Units (urad). Pointer to a double. Command to FSM electronics.

*/

// #include "mex.h"  /* mexPrintf("P_hat_kp[0][0] = %e \n", P_hat_kp[0][0]); */
// #include <time.h>
#include "CF_LOS_FB_40Hz.h"

#include <math.h>
#include <string.h>

void CF_LOS_FB_function(double *U_X, double *U_Y, const int PC_onoff_flag,
                        const int HOLD_flag, const double LOS_CMD_X,
                        const double LOS_CMD_Y, const double LOS_MEAS_X,
                        const double LOS_MEAS_Y)
{
  //    clock_t begin, end;
  //    double time_spent = 0.0;

  //    begin = clock();

  int i = 0, j = 0;

  static double x_fb_x_k[N_pc_fb_x] = {0.0, 0.0, 0.0, 0.0,
                                       0.0}; /* feedback states */
  static double x_fb_y_k[N_pc_fb_y] = {0.0, 0.0, 0.0, 0.0, 0.0};
  double x_fb_x_kp1[N_pc_fb_x];
  double x_fb_y_kp1[N_pc_fb_y];

  double ERROR_X;
  double ERROR_Y;
  double U_X_out;
  double U_Y_out;

  static double previous_U_X = 0.0;
  static double previous_U_Y = 0.0;

  if (HOLD_flag) {
    U_X[0] = previous_U_X;
    U_Y[0] = previous_U_Y;
  } else {
    if (PC_onoff_flag) {
      ERROR_X = (LOS_CMD_X - LOS_MEAS_X);
      ERROR_Y = (LOS_CMD_Y - LOS_MEAS_Y);

      /* X channel feedback compensator */
      for (i = 0; i < N_pc_fb_x; i++) {
        x_fb_x_kp1[i] = 0.0;
        for (j = 0; j < N_pc_fb_x; j++) {
          x_fb_x_kp1[i] = x_fb_x_kp1[i] + A_pc_fb_x_d[i][j] * x_fb_x_k[j];
        }
        x_fb_x_kp1[i] = x_fb_x_kp1[i] + B_pc_fb_x_d[i] * ERROR_X;
      }

      U_X_out = 0.0;
      for (i = 0; i < N_pc_fb_x; i++) {
        U_X_out = U_X_out + C_pc_fb_x_d[i] * x_fb_x_k[i];
      }
      U_X_out = U_X_out + D_pc_fb_x_d * ERROR_X;

      for (i = 0; i < N_pc_fb_x; i++) /* Get ready for next iteration by passing
                                         new state to old state. */
      {
        x_fb_x_k[i] = x_fb_x_kp1[i];
      }

      /* Y channel feedback compensator */
      for (i = 0; i < N_pc_fb_y; i++) {
        x_fb_y_kp1[i] = 0.0;
        for (j = 0; j < N_pc_fb_y; j++) {
          x_fb_y_kp1[i] = x_fb_y_kp1[i] + A_pc_fb_y_d[i][j] * x_fb_y_k[j];
        }
        x_fb_y_kp1[i] = x_fb_y_kp1[i] + B_pc_fb_y_d[i] * ERROR_Y;
      }

      U_Y_out = 0.0;
      for (i = 0; i < N_pc_fb_y; i++) {
        U_Y_out = U_Y_out + C_pc_fb_y_d[i] * x_fb_y_k[i];
      }
      U_Y_out = U_Y_out + D_pc_fb_y_d * ERROR_Y;

      for (i = 0; i < N_pc_fb_y; i++) /* Get ready for next iteration by passing
                                         new state to old state. */
      {
        x_fb_y_k[i] = x_fb_y_kp1[i];
      }

      U_X[0] = U_X_out;
      U_Y[0] = U_Y_out;
      previous_U_X = U_X_out;
      previous_U_Y = U_Y_out;
    } else {
      memset(x_fb_x_k, 0, sizeof(x_fb_x_k));
      memset(x_fb_y_k, 0, sizeof(x_fb_y_k));

      U_X[0] = 0.0;
      U_Y[0] = 0.0;
      previous_U_X = 0.0;
      previous_U_Y = 0.0;
    }
  }

  //    end = clock();
  //    time_spent = (double) (end - begin)/CLOCKS_PER_SEC;

  //    mexPrintf("time_spent = %e \n", time_spent);
}
