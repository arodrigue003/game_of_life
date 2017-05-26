
#ifndef COMPUTE_IS_DEF
#define COMPUTE_IS_DEF

typedef void (*void_func_t) (void);
typedef unsigned (*int_func_t) (unsigned);

extern void_func_t first_touch [];
extern int_func_t compute [];
extern char *version_name [];
extern unsigned opencl_used [];
extern unsigned init_required [];
extern void_func_t init_version [];
extern void_func_t free_version [];

extern unsigned version;


#endif
