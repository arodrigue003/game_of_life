#ifndef OCL_IS_DEF
#define OCL_IS_DEF


#include <SDL_opengl.h>

void ocl_init (void);
void ocl_map_textures (GLuint texid);
void ocl_send_image (unsigned *image);
unsigned ocl_compute (unsigned nb_iter);
void ocl_wait (void);
void ocl_update_texture (void);
unsigned compute_ratio(int gpu_frac);
unsigned ocl_compute_hybrid(unsigned nb_iter, int nb_tranches);
unsigned ocl_compute_opt (unsigned nb_iter);
void get_picture_back(unsigned* picture);
void put_picture(unsigned *next_img);

extern unsigned SIZE, TILE;

#endif
