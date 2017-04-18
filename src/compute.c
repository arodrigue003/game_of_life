
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
unsigned openMP_for_v0 (unsigned nb_iter);
unsigned openMP_for_v1 (unsigned nb_iter);
unsigned openMP_for_v2 (unsigned nb_iter);
unsigned compute_v3 (unsigned nb_iter);


void_func_t first_touch [] = {
    NULL,
    first_touch_v1,
    first_touch_v2,
    NULL,
    NULL,
    NULL,
    NULL,
};

int_func_t compute [] = {
    compute_v0,
    compute_v1,
    compute_v2,
    openMP_for_v0,
    openMP_for_v1,
    openMP_for_v2,
    compute_v3,
};

char *version_name [] = {
    "Séquentielle",
    "Séquentielle tuilée",
    "Séquentielle optimisé",
    "OpenMP",
    "OpenMP tuilé",
    "OpenMp optimisé",
    "OpenCL",
};

unsigned opencl_used [] = {
    0,
    0,
    0,
    0,
    0,
    0,
    1,
};

///////////////////////////// Version séquentielle simple


unsigned compute_v0(unsigned nb_iter) {

    for (unsigned it = 1; it <= nb_iter; it ++) {
        for (int x = 1; x < DIM-1; x++) {
            for (int y = 1; y < DIM-1; y++) {
                int n = (cur_img(x-1, y-1) !=0) +
                        (cur_img(x-1, y)   !=0) +
                        (cur_img(x-1, y+1) !=0) +
                        (cur_img(x  , y-1) !=0) +
                        (cur_img(x  , y+1) !=0) +
                        (cur_img(x+1, y-1) !=0) +
                        (cur_img(x+1, y)   !=0) +
                        (cur_img(x+1, y+1) !=0);
                if (cur_img(x, y)) {
                    if (n>=2 && n<=3)
                        next_img(x,y) = couleur;
                    else
                        next_img(x,y) = 0;
                }
                else {
                    if (n==3)
                        next_img(x,y) = couleur;
                    else
                        next_img(x,y) = 0;
                }

            }
        }
        swap_images();
    }
    // retourne le nombre d'étapes nécessaires à la
    // stabilisation du calcul ou bien 0 si le calcul n'est pas
    // stabilisé au bout des nb_iter itérations
    return 0;
}


///////////////////////////// Version OpenMP de base

unsigned openMP_for_v0(unsigned nb_iter) {
    for (unsigned it = 1; it <= nb_iter; it ++) {

#pragma omp parallel for schedule(guided,4) collapse(2)
        for (int x = 1; x < DIM-1; x++) {
            for (int y = 1; y < DIM-1; y++) {
                int n = (cur_img(x-1, y-1) !=0) +
                        (cur_img(x-1, y)   !=0) +
                        (cur_img(x-1, y+1) !=0) +
                        (cur_img(x  , y-1) !=0) +
                        (cur_img(x  , y+1) !=0) +
                        (cur_img(x+1, y-1) !=0) +
                        (cur_img(x+1, y)   !=0) +
                        (cur_img(x+1, y+1) !=0);
                if (cur_img(x, y)) {
                    if (n>=2 && n<=3)
                        next_img(x,y) = couleur;
                    else
                        next_img(x,y) = 0;
                }
                else {
                    if (n==3)
                        next_img(x,y) = couleur;
                    else
                        next_img(x,y) = 0;
                }

            }
        }
        swap_images();
    }

    return 0;
}


void first_touch_v1 (void) {}

///////////////////////////// Version séquentielle avec tuiles

#define GRAIN 32

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v1(unsigned nb_iter){
    unsigned tranche = DIM / GRAIN;

    for (unsigned it = 1; it <= nb_iter; it ++) {
        for (unsigned tuilex = 0; tuilex < tranche; tuilex++) {
            for (unsigned tuiley = 0; tuiley < tranche; tuiley++) {
                for (int xloc = 0; xloc < GRAIN; xloc++) {
                    for (int yloc = 0; yloc < GRAIN; yloc++) {
                        unsigned x=tuilex*GRAIN+xloc;
                        unsigned y=tuiley*GRAIN+yloc;
                        if (x>0 && x<DIM && y>0 && y<DIM) {
                            int n = (cur_img(x-1, y-1) !=0) +
                                    (cur_img(x-1, y)   !=0) +
                                    (cur_img(x-1, y+1) !=0) +
                                    (cur_img(x  , y-1) !=0) +
                                    (cur_img(x  , y+1) !=0) +
                                    (cur_img(x+1, y-1) !=0) +
                                    (cur_img(x+1, y)   !=0) +
                                    (cur_img(x+1, y+1) !=0);
                            if (cur_img(x, y)) {
                                if (n>=2 && n<=3)
                                    next_img(x,y) = couleur;
                                else
                                    next_img(x,y) = 0;
                            }
                            else {
                                if (n==3)
                                    next_img(x,y) = couleur;
                                else
                                    next_img(x,y) = 0;
                            }
                        }
                    }
                }
            }
        }
        swap_images();
    }

    return 0;
}


///////////////////////////// Version OpenMP avec tuiles

unsigned openMP_for_v1(unsigned nb_iter) {
    unsigned tranche = DIM / GRAIN;

    for (unsigned it = 1; it <= nb_iter; it ++) {

        #pragma omp parallel for schedule(guided,4) collapse(2)
        for (unsigned tuilex = 0; tuilex < tranche; tuilex++) {
            for (unsigned tuiley = 0; tuiley < tranche; tuiley++) {

                for (int xloc = 0; xloc < GRAIN; xloc++) {
                    for (int yloc = 0; yloc < GRAIN; yloc++) {
                        unsigned x=tuilex*GRAIN+xloc;
                        unsigned y=tuiley*GRAIN+yloc;
                        if (x>0 && x<DIM && y>0 && y<DIM) {
                            int n = (cur_img(x-1, y-1) !=0) +
                                    (cur_img(x-1, y)   !=0) +
                                    (cur_img(x-1, y+1) !=0) +
                                    (cur_img(x  , y-1) !=0) +
                                    (cur_img(x  , y+1) !=0) +
                                    (cur_img(x+1, y-1) !=0) +
                                    (cur_img(x+1, y)   !=0) +
                                    (cur_img(x+1, y+1) !=0);
                            if (cur_img(x, y)) {
                                if (n>=2 && n<=3)
                                    next_img(x,y) = couleur;
                                else
                                    next_img(x,y) = 0;
                            }
                            else {
                                if (n==3)
                                    next_img(x,y) = couleur;
                                else
                                    next_img(x,y) = 0;
                            }
                        }
                    }
                }

            }
        }


        swap_images();
    }

    return 0;
}

///////////////////////////// Version séquentielle optimisé

#define coord(x,y) (x+1)*tranche+y+1
bool* curr_unchanged;
bool* next_unchanged;
bool* unchangedTop;
bool* unchangedRight;
bool* unchangedBottom;
bool* unchangedLeft;

void swap_tiles() {
    bool* tmp = curr_unchanged;

    curr_unchanged = next_unchanged;
    next_unchanged = tmp;
}

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v2(unsigned nb_iter)
{
    unsigned tranche = DIM / GRAIN;

    static bool first = true;
    if (first) {
        first = false;
        //greater size in order to don't test if a tile is en a extreme position or not
        curr_unchanged = malloc(sizeof(unsigned)*(tranche+2)*(tranche+2));
        for(int tuilex=-1; tuilex<=tranche;tuilex++)
            for (int tuiley=-1; tuiley<=tranche; tuiley++)
                curr_unchanged[coord(tuilex,tuiley)] = false;
        next_unchanged = malloc(sizeof(unsigned)*(tranche+2)*(tranche+2));
        for(int tuilex=-1; tuilex<=tranche;tuilex++)
            for (int tuiley=-1; tuiley<=tranche; tuiley++)
                next_unchanged[coord(tuilex,tuiley)] = false;
    }


    for (unsigned it = 1; it <= nb_iter; it ++) {
        for (int tuilex = 0; tuilex < tranche; tuilex++) {
            for (int tuiley = 0; tuiley < tranche; tuiley++) {

                if (!curr_unchanged[coord(tuilex,tuiley)] ||
                        !curr_unchanged[coord(tuilex+1,tuiley)] ||
                        !curr_unchanged[coord(tuilex-1,tuiley)] ||
                        !curr_unchanged[coord(tuilex,tuiley+1)] ||
                        !curr_unchanged[coord(tuilex,tuiley-1)] ||
                        !curr_unchanged[coord(tuilex+1,tuiley+1)] ||
                        !curr_unchanged[coord(tuilex+1,tuiley-1)] ||
                        !curr_unchanged[coord(tuilex-1,tuiley+1)] ||
                        !curr_unchanged[coord(tuilex-1,tuiley-1)]) {
                    bool tuile_unchanged = true;
                    for (int xloc = 0; xloc < GRAIN; xloc++) {
                        for (int yloc = 0; yloc < GRAIN; yloc++) {
                            unsigned x=tuilex*GRAIN+xloc;
                            unsigned y=tuiley*GRAIN+yloc;
                            if (x>0 && x<DIM && y>0 && y<DIM) {
                                int n = (cur_img(x-1, y-1) !=0) +
                                        (cur_img(x-1, y)   !=0) +
                                        (cur_img(x-1, y+1) !=0) +
                                        (cur_img(x  , y-1) !=0) +
                                        (cur_img(x  , y+1) !=0) +
                                        (cur_img(x+1, y-1) !=0) +
                                        (cur_img(x+1, y)   !=0) +
                                        (cur_img(x+1, y+1) !=0);
                                if (cur_img(x, y)) {
                                    if (n>=2 && n<=3)
                                        next_img(x,y) = couleur;
                                    else {
                                        //cell dies
                                        tuile_unchanged = false;
                                        next_img(x,y) = 0;
                                    }
                                }
                                else {
                                    if (n==3) {
                                        //cell creation
                                        tuile_unchanged = false;
                                        next_img(x,y) = couleur;
                                    }
                                        else
                                        next_img(x,y) = 0;
                                }
                            }
                        }
                    }
                    next_unchanged[coord(tuilex,tuiley)] = tuile_unchanged;
                }
                else {
                    next_unchanged[coord(tuilex,tuiley)] = true;
                }

            }
        }
        swap_tiles();
        swap_images();
    }

    return 0;
}

///////////////////////////// Version OpenMP optimisée

unsigned openMP_for_v2(unsigned nb_iter)
{
    unsigned tranche = DIM / GRAIN;

    static bool first = true;
    if (first) {
        first = false;
        //greater size in order to don't test if a tile is en a extreme position or not
        curr_unchanged = malloc(sizeof(unsigned)*(tranche+2)*(tranche+2));
        for(int tuilex=-1; tuilex<=tranche;tuilex++)
            for (int tuiley=-1; tuiley<=tranche; tuiley++)
                curr_unchanged[coord(tuilex,tuiley)] = false;
        next_unchanged = malloc(sizeof(unsigned)*(tranche+2)*(tranche+2));
        for(int tuilex=-1; tuilex<=tranche;tuilex++)
            for (int tuiley=-1; tuiley<=tranche; tuiley++)
                next_unchanged[coord(tuilex,tuiley)] = false;
    }


    for (unsigned it = 1; it <= nb_iter; it ++) {

        #pragma omp parallel for schedule(guided,4) collapse(2)
        for (int tuilex = 0; tuilex < tranche; tuilex++) {
            for (int tuiley = 0; tuiley < tranche; tuiley++) {

                if (!curr_unchanged[coord(tuilex,tuiley)] ||
                        !curr_unchanged[coord(tuilex+1,tuiley)] ||
                        !curr_unchanged[coord(tuilex-1,tuiley)] ||
                        !curr_unchanged[coord(tuilex,tuiley+1)] ||
                        !curr_unchanged[coord(tuilex,tuiley-1)] ||
                        !curr_unchanged[coord(tuilex+1,tuiley+1)] ||
                        !curr_unchanged[coord(tuilex+1,tuiley-1)] ||
                        !curr_unchanged[coord(tuilex-1,tuiley+1)] ||
                        !curr_unchanged[coord(tuilex-1,tuiley-1)]) {
                    bool tuile_unchanged = true;
                    for (int xloc = 0; xloc < GRAIN; xloc++) {
                        for (int yloc = 0; yloc < GRAIN; yloc++) {
                            unsigned x=tuilex*GRAIN+xloc;
                            unsigned y=tuiley*GRAIN+yloc;
                            if (x>0 && x<DIM && y>0 && y<DIM) {
                                int n = (cur_img(x-1, y-1) !=0) +
                                        (cur_img(x-1, y)   !=0) +
                                        (cur_img(x-1, y+1) !=0) +
                                        (cur_img(x  , y-1) !=0) +
                                        (cur_img(x  , y+1) !=0) +
                                        (cur_img(x+1, y-1) !=0) +
                                        (cur_img(x+1, y)   !=0) +
                                        (cur_img(x+1, y+1) !=0);
                                if (cur_img(x, y)) {
                                    if (n>=2 && n<=3)
                                        next_img(x,y) = couleur;
                                    else {
                                        //cell dies
                                        tuile_unchanged = false;
                                        next_img(x,y) = 0;
                                    }
                                }
                                else {
                                    if (n==3) {
                                        //cell creation
                                        tuile_unchanged = false;
                                        next_img(x,y) = couleur;
                                    }
                                        else
                                        next_img(x,y) = 0;
                                }
                            }
                        }
                    }
                    next_unchanged[coord(tuilex,tuiley)] = tuile_unchanged;
                }
                else {
                    next_unchanged[coord(tuilex,tuiley)] = true;
                }

            }
        }


        swap_tiles();
        swap_images();
    }

    return 0;
}




void first_touch_v2 ()
{

}

///////////////////////////// Version OpenCL

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v3 (unsigned nb_iter)
{
    return ocl_compute (nb_iter);
}
