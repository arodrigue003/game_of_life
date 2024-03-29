
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdbool.h>
#include <omp.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <OpenGL/CGLContext.h>
#include <OpenGL/CGLCurrent.h>
#else
#include <CL/opencl.h>
#include <GL/glx.h>
#endif

#include "constants.h"
#include "error.h"
#include "ocl.h"
#include "graphics.h"
#include "debug.h"


#define check(err, ...)					\
    do {							\
    if (err != CL_SUCCESS) {				\
    fprintf (stderr, "(%d) Error: " __VA_ARGS__ "\n", err);	\
    exit (EXIT_FAILURE);				\
    }							\
    } while (0)

#define MAX_PLATFORMS 3
#define MAX_DEVICES   5

unsigned TILEX = 16;
unsigned TILEY = 16;
unsigned SIZE = 0;

static char *kernel_name = "scrollup";

cl_int err;
cl_context context;
cl_kernel update_kernel;
cl_kernel compute_kernel;
cl_command_queue queue;
cl_mem tex_buffer, cur_buffer, next_buffer;
cl_mem curr_unchanged, next_unchanged;

static size_t file_size (const char *filename)
{
    struct stat sb;

    if (stat (filename, &sb) < 0) {
        perror ("stat");
        abort ();
    }
    return sb.st_size;
}

static char *file_load (const char *filename)
{
    FILE *f;
    char *b;
    size_t s;
    size_t r;

    s = file_size (filename);
    b = malloc (s+1);
    if (!b) {
        perror ("malloc");
        exit (1);
    }
    f = fopen (filename, "r");
    if (f == NULL) {
        perror ("fopen");
        exit (1);
    }
    r = fread (b, s, 1, f);
    if (r != 1) {
        perror ("fread");
        exit (1);
    }
    b[s] = '\0';
    return b;
}

static void ocl_acquire (void)
{
    cl_int err;

    err = clEnqueueAcquireGLObjects (queue, 1, &tex_buffer, 0, NULL, NULL);
    check (err, "Failed to acquire lock");
}

static void ocl_release (void)
{
    cl_int err;

    err = clEnqueueReleaseGLObjects (queue, 1, &tex_buffer, 0, NULL, NULL);
    check (err, "Failed to release lock");
}

void ocl_init (void)
{
    char name [1024], vendor [1024];
    cl_platform_id pf [MAX_PLATFORMS];
    cl_uint nb_platforms = 0;
    cl_device_id devices [MAX_DEVICES];
    cl_program program;                 // compute program
    cl_device_type dtype;
    cl_uint nb_devices = 0;
    char *str = NULL;
    unsigned platform_no = 0;
    unsigned dev = 0;

    str = getenv ("PLATFORM");
    if (str != NULL)
        platform_no = atoi (str);

    str = getenv ("DEVICE");
    if (str != NULL)
        dev = atoi (str);

    str = getenv ("SIZE");
    if (str != NULL)
        SIZE = atoi (str);
    else
        SIZE = DIM;

    str = getenv ("TILEX");
    if (str != NULL)
        TILEX = atoi (str);
    else
        TILEX = 16;

    str = getenv ("TILEY");
    if (str != NULL)
        TILEY = atoi (str);
    else
        TILEY = TILEX;

    str = getenv ("KERNEL");
    if (str != NULL)
        kernel_name = str;

    if (SIZE > DIM)
        exit_with_error ("SIZE (%d) cannot exceed DIM (%d)", SIZE, DIM);

    // Get list of OpenCL platforms detected
    //
    err = clGetPlatformIDs (MAX_PLATFORMS, pf, &nb_platforms);
    check (err, "Failed to get platform IDs");

    PRINT_DEBUG ('o', "%d OpenCL platforms detected:\n", nb_platforms);

    if (platform_no >= nb_platforms)
        exit_with_error ("Platform number #%d too high\n", platform_no);
    
    err = clGetPlatformInfo (pf [platform_no], CL_PLATFORM_NAME, 1024, name, NULL);
    check (err, "Failed to get Platform Info");

    err = clGetPlatformInfo (pf [platform_no], CL_PLATFORM_VENDOR, 1024, vendor, NULL);
    check (err, "Failed to get Platform Info");

    printf ("Using platform %d: %s - %s\n", platform_no, name, vendor);

    // Get list of devices
    //
    err = clGetDeviceIDs (pf [platform_no], CL_DEVICE_TYPE_GPU,
                          MAX_DEVICES, devices, &nb_devices);
    PRINT_DEBUG ('o', "nb devices = %d\n", nb_devices);

    if (nb_devices == 0) {
        exit_with_error ("No GPU found on platform %d (%s - %s). Try PLATFORM=<p> ./prog blabla\n",
                         platform_no, name, vendor);
    }
    if (dev >= nb_devices)
        exit_with_error ("Device number #%d too high\n", dev);

    err = clGetDeviceInfo (devices [dev], CL_DEVICE_NAME, 1024, name, NULL);
    check (err, "Cannot get type of device");
    
    err = clGetDeviceInfo (devices [dev], CL_DEVICE_TYPE, sizeof (cl_device_type), &dtype, NULL);
    check (err, "Cannot get type of device");

    printf ("Using Device %d : %s [%s]\n", dev, (dtype == CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU", name);

    if (graphics_display_enabled ()) {
#ifdef __APPLE__
        CGLContextObj cgl_context = CGLGetCurrentContext ();
        CGLShareGroupObj sharegroup = CGLGetShareGroup (cgl_context);
        cl_context_properties properties [] = {
            CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
            (cl_context_properties) sharegroup,
            0
        };
#else
        cl_context_properties properties [] = {
            CL_GL_CONTEXT_KHR,
            (cl_context_properties) glXGetCurrentContext (),
            CL_GLX_DISPLAY_KHR,
            (cl_context_properties) glXGetCurrentDisplay (),
            CL_CONTEXT_PLATFORM, (cl_context_properties) pf [platform_no],
            0
        };
#endif

        context = clCreateContext (properties, 1, &devices [dev], NULL, NULL, &err);
    } else
        context = clCreateContext (NULL, 1, &devices [dev], NULL, NULL, &err);

    check (err, "Failed to create compute context");

    // Load program source into memory
    //
    const char	*opencl_prog;
    opencl_prog = file_load ("kernel/compute.cl");

    // Attach program source to context
    //
    program = clCreateProgramWithSource (context, 1, &opencl_prog, NULL, &err);
    check (err, "Failed to create program");

    // Compile program
    //
    {
        char flags[1024];

        sprintf (flags,
                 "-cl-mad-enable -cl-fast-relaxed-math -DDIM=%d -DSIZE=%d -DTILEX=%d -DTILEY=%d",
                 DIM, SIZE, TILEX, TILEY);

        err = clBuildProgram (program, 0, NULL, flags, NULL, NULL);
        if(err != CL_SUCCESS) {
            size_t len;

            // Display compiler log
            //
            clGetProgramBuildInfo (program, devices [dev], CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
            {
                char buffer[len+1];

                fprintf (stderr, "--- Compiler log ---\n");
                clGetProgramBuildInfo (program, devices [dev], CL_PROGRAM_BUILD_LOG,
                                       sizeof (buffer), buffer, NULL);
                fprintf (stderr, "%s\n", buffer);
                fprintf (stderr, "--------------------\n");
            }
            if(err != CL_SUCCESS)
                exit_with_error ("Failed to build program!\n");
        }
    }

    // Create the compute kernel in the program we wish to run
    //
    compute_kernel = clCreateKernel (program, kernel_name, &err);
    check (err, "Failed to create compute kernel");

    printf ("Using kernel: %s\n", kernel_name);

    update_kernel = clCreateKernel (program, "update_texture", &err);
    check (err, "Failed to create compute kernel");

    // Create a command queue
    //
    queue = clCreateCommandQueue (context, devices [dev], CL_QUEUE_PROFILING_ENABLE, &err);
    check (err,"Failed to create command queue");

    // Allocate buffers inside device memory
    //
    cur_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, sizeof(unsigned) * DIM * DIM,
                                 NULL, NULL);
    if (!cur_buffer)
        exit_with_error ("Failed to allocate input buffer");

    next_buffer = clCreateBuffer (context, CL_MEM_READ_WRITE, sizeof(unsigned) * DIM * DIM,
                                  NULL, NULL);
    if (!next_buffer)
        exit_with_error ("Failed to allocate output buffer");


    unsigned tranche = DIM / TILEX;

    curr_unchanged = clCreateBuffer (context, CL_MEM_READ_WRITE, sizeof(bool) * (tranche+2) * (tranche+2),
                                     NULL, NULL);
    if (!curr_unchanged)
        exit_with_error ("Failed to allocate input buffer");

    next_unchanged = clCreateBuffer (context, CL_MEM_READ_WRITE, sizeof(bool) * (tranche+2) * (tranche+2),
                                     NULL, NULL);
    if (!next_unchanged)
        exit_with_error ("Failed to allocate output buffer");

    printf ("Using %dx%d workitems grouped in %dx%d tiles \n", SIZE, SIZE, TILEX, TILEY);

}

void ocl_map_textures (GLuint texid)
{
    /* Shared texture buffer with OpenGL. */
    tex_buffer = clCreateFromGLTexture (context, CL_MEM_READ_WRITE,
                                        GL_TEXTURE_2D, 0, texid, &err);
    check (err, "Failed to map texture buffer\n");
}

#define coord(x,y) (x+1)*tranche+y+1

void ocl_send_image (unsigned *image)
{

    err = clEnqueueWriteBuffer (queue, cur_buffer, CL_TRUE, 0,
                                sizeof (unsigned) * DIM * DIM, image, 0, NULL, NULL);
    check (err, "Failed to write to cur_buffer");

    err = clEnqueueWriteBuffer (queue, next_buffer, CL_TRUE, 0,
                                sizeof (unsigned) * DIM * DIM, image, 0, NULL, NULL);
    check (err, "Failed to write to next_buffer");


    PRINT_DEBUG ('o', "Initial image sent to device.\n");

    unsigned tranche = DIM / TILEX;
    bool *unchanged = malloc(sizeof(unsigned)*(tranche+2)*(tranche+2));
    for(int tuilex=-1; tuilex<=tranche;tuilex++)
        for (int tuiley=-1; tuiley<=tranche; tuiley++)
            unchanged[coord(tuilex,tuiley)] = false;

    err = clEnqueueWriteBuffer (queue, curr_unchanged, CL_TRUE, 0,
                                sizeof(bool) * (tranche+2) * (tranche+2), unchanged, 0, NULL, NULL);
    check (err, "Failed to write to cur_buffer");

    err = clEnqueueWriteBuffer (queue, next_unchanged, CL_TRUE, 0,
                                sizeof(bool) * (tranche+2) * (tranche+2), unchanged, 0, NULL, NULL);
    check (err, "Failed to write to next_buffer");
    free(unchanged);

}

unsigned ocl_compute (unsigned nb_iter)
{
    size_t global[2] = { SIZE, SIZE };  // global domain size for our calculation
    size_t local[2]  = { TILEX, TILEY };  // local domain size for our calculation

    for (unsigned it = 1; it <= nb_iter; it ++) {

        // Set kernel arguments
        //
        err = 0;
        err  = clSetKernelArg (compute_kernel, 0, sizeof (cl_mem), &cur_buffer);
        err  = clSetKernelArg (compute_kernel, 1, sizeof (cl_mem), &next_buffer);
        check (err, "Failed to set kernel arguments");

        err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, local,
                                      0, NULL, NULL);
        check(err, "Failed to execute kernel");

        // Swap buffers
        { cl_mem tmp = cur_buffer; cur_buffer = next_buffer; next_buffer = tmp; }

    }

    return 0;
}

unsigned ocl_compute_opt (unsigned nb_iter) {

    size_t global[2] = { SIZE, SIZE };  // global domain size for our calculation
    size_t local[2]  = { TILEX, TILEY };  // local domain size for our calculation

    for (unsigned it = 1; it <= nb_iter; it ++) {

        // Set kernel arguments
        //
        err = 0;
        err = clSetKernelArg (compute_kernel, 0, sizeof (cl_mem), &cur_buffer);
        err = clSetKernelArg (compute_kernel, 1, sizeof (cl_mem), &next_buffer);
        err = clSetKernelArg (compute_kernel, 2, sizeof (cl_mem), &curr_unchanged);
        err = clSetKernelArg (compute_kernel, 3, sizeof (cl_mem), &next_unchanged);
        check (err, "Failed to set kernel arguments");

        err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, local,
                                      0, NULL, NULL);
        check(err, "Failed to execute kernel");

        // Swap buffers
        { cl_mem tmp = cur_buffer; cur_buffer = next_buffer; next_buffer = tmp;
            tmp = curr_unchanged; curr_unchanged = next_unchanged; next_unchanged = tmp;}


    }

    return 0;
}

unsigned compute_ratio(int gpu_frac) {
    int tranche = SIZE/TILEY;
    int nb_tranches = tranche - (int) tranche*(100-gpu_frac)/100;
    return nb_tranches*TILEY;
}

void get_picture_back(unsigned* picture) {
    err = clEnqueueReadBuffer (queue, cur_buffer, CL_TRUE, 0,
                                sizeof(unsigned) * DIM * DIM, picture, 0, NULL, NULL);
    check (err, "Failed to write to cur_buffer");
}

void put_picture(unsigned *picture) {
    err = clEnqueueWriteBuffer (queue, cur_buffer, CL_TRUE, 0,
                                sizeof(unsigned) * DIM * DIM, picture, 0, NULL, NULL);
    check (err, "Failed to write to cur_buffer");
}

unsigned ocl_compute_hybrid(unsigned nb_iter, int nb_tranches) {

    size_t global[2] = { SIZE, nb_tranches };  // global domain size for our calculation
    size_t local[2]  = { TILEX, TILEY };  // local domain size for our calculation

    for (unsigned it = 1; it <= nb_iter; it ++) {

        // Set kernel arguments
        //
        err = 0;
        err  = clSetKernelArg (compute_kernel, 0, sizeof (cl_mem), &cur_buffer);
        err  = clSetKernelArg (compute_kernel, 1, sizeof (cl_mem), &next_buffer);
        check (err, "Failed to set kernel arguments");

        err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, local,
                                      0, NULL, NULL);
        check(err, "Failed to execute kernel");
        { cl_mem tmp = cur_buffer; cur_buffer = next_buffer; next_buffer = tmp; }

    }

    return 0;
}


void ocl_wait (void)
{
    // Wait for the command commands to get serviced before reading back results
    //
    clFinish (queue);
}

void ocl_update_texture (void)
{
    size_t global[2] = { DIM, DIM };  // global domain size for our calculation
    size_t local[2]  = { 16, 16 };  // local domain size for our calculation

    ocl_acquire ();

    // Set kernel arguments
    //
    err = 0;
    err  = clSetKernelArg (update_kernel, 0, sizeof (cl_mem), &cur_buffer);
    err  = clSetKernelArg (update_kernel, 1, sizeof (cl_mem), &tex_buffer);
    check (err, "Failed to set kernel arguments");

    err = clEnqueueNDRangeKernel (queue, update_kernel, 2, NULL, global, local,
                                  0, NULL, NULL);
    check(err, "Failed to execute kernel");

    ocl_release ();

    clFinish (queue);
}
