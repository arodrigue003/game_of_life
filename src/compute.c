
#include "compute.h"
#include "graphics.h"
#include "debug.h"
#include "ocl.h"

#include <stdbool.h>
#include <stdio.h>

static unsigned couleur = 0xFFFF00FF; // Yellow

unsigned version = 0;

void first_touch_v1 (void);
void first_touch_v2 (void);

unsigned compute_v0 (unsigned nb_iter);
unsigned compute_v1 (unsigned nb_iter);
unsigned compute_v2 (unsigned nb_iter);
unsigned compute_v3 (unsigned nb_iter);

void_func_t first_touch [] = {
    NULL,
    first_touch_v1,
    first_touch_v2,
    NULL,
};

int_func_t compute [] = {
    compute_v0,
    compute_v1,
    compute_v2,
    compute_v3,
};

char *version_name [] = {
    "Séquentielle",
    "OpenMP",
    "OpenMP zone",
    "OpenCL",
};

unsigned opencl_used [] = {
    0,
    0,
    0,
    1,
};

///////////////////////////// Version séquentielle simple


unsigned compute_v0 (unsigned nb_iter)
{

    for (unsigned it = 1; it <= nb_iter; it ++) {
        for (int i = 1; i < DIM-1; i++) {
            for (int j = 1; j < DIM-1; j++) {
                int n = (cur_img(i-1, j-1) !=0) +
                        (cur_img(i-1, j)   !=0) +
                        (cur_img(i-1, j+1) !=0) +
                        (cur_img(i  , j-1) !=0) +
                        (cur_img(i  , j+1) !=0) +
                        (cur_img(i+1, j-1) !=0) +
                        (cur_img(i+1, j)   !=0) +
                        (cur_img(i+1, j+1) !=0);
                if (cur_img(i, j)) {
                    if (n>=2 && n<=3)
                        next_img(i,j) = couleur;
                    else
                        next_img(i,j) = 0;
                }
                else {
                    if (n==3)
                        next_img(i,j) = couleur;
                    else
                        next_img(i,j) = 0;
                }

            }
        }
        swap_images ();
    }
    // retourne le nombre d'étapes nécessaires à la
    // stabilisation du calcul ou bien 0 si le calcul n'est pas
    // stabilisé au bout des nb_iter itérations
    return 0;
}


///////////////////////////// Version OpenMP de base

void first_touch_v1 ()
{
    int i,j ;

#pragma omp parallel for
    for(i=0; i<DIM ; i++) {
        for(j=0; j < DIM ; j += 512)
            next_img (i, j) = cur_img (i, j) = 0 ;
    }
}

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v1(unsigned nb_iter)
{
    return 0;
}



///////////////////////////// Version OpenMP optimisée

void first_touch_v2 ()
{

}

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v2(unsigned nb_iter)
{
    return 0; // on ne s'arrête jamais
}


///////////////////////////// Version OpenCL

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v3 (unsigned nb_iter)
{
    return ocl_compute (nb_iter);
}
