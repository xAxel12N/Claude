/* sensors.h - hardware temperatures, fans and load for the GUI. */
#ifndef SENSORS_H
#define SENSORS_H

enum { SENSOR_TEMP, SENSOR_FAN, SENSOR_PERCENT };

#define SENSOR_MAX 24
typedef struct {
    int kind;      /* SENSOR_TEMP in C, SENSOR_FAN in RPM, SENSOR_PERCENT */
    double value;
    char label[48];
} Sensor;

typedef struct {
    Sensor s[SENSOR_MAX];
    int n, ready;
    double cpu;        /* CPU load in percent, -1 until known */
    char gpu_name[64]; /* NVIDIA card name, if nvidia-smi works */
} SensorSnap;

void sensors_start(void);
void sensors_get(SensorSnap *out);

#endif
