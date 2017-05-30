
#include "compute.h"
#include "graphics.h"
#include "debug.h"
#include "ocl.h"

#include <stdbool.h>
#include <stdio.h>

#define GRAIN 16

static unsigned couleur = 0xFFFF00FF; // Yellow

unsigned version = 0;

unsigned compute_v0 (unsigned nb_iter); //l99
unsigned compute_v1 (unsigned nb_iter);
void init_compute_tuile_opt();
unsigned compute_v2 (unsigned nb_iter);
void free_compute_tuile_opt();
unsigned openMP_for_v0 (unsigned nb_iter); //l118
unsigned openMP_for_v1 (unsigned nb_iter);
unsigned openMP_for_v2 (unsigned nb_iter);
unsigned openMP_task_v1(unsigned nb_iter);
unsigned openMP_task_v2(unsigned nb_iter);
unsigned compute_v3 (unsigned nb_iter);
unsigned compute_v4 (unsigned nb_iter);
unsigned compute_v5 (unsigned nb_iter);

void_func_t first_touch [] = {
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL
};

int_func_t compute [] = {
    compute_v0,
    compute_v1,
    compute_v2,
    openMP_for_v0,
    openMP_for_v1,
    openMP_for_v2,
    openMP_task_v1,
    openMP_task_v2,
    compute_v3,
    compute_v4,
    compute_v5
};

char *version_name [] = {
    "Séquentielle",
    "Séquentielle tuilée",
    "Séquentielle optimisé",
    "OpenMP for",
    "OpenMP for tuilé",
    "OpenMp for optimisé",
    "OpenMP task tuilé",
    "OpenMP task optimisé",
    "OpenCL",
    "OpenCL optimisé",
    "Hybrid"
};

unsigned opencl_used [] = {
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1
};

unsigned init_required [] = {
    0,
    0,
    1,
    0,
    0,
    1,
    0,
    1,
    0,
    0,
    0
};

void_func_t init_version [] = {
    NULL,
    NULL,
    init_compute_tuile_opt,
    NULL,
    NULL,
    init_compute_tuile_opt,
    NULL,
    init_compute_tuile_opt,
    NULL,
    NULL,
    NULL
};

void_func_t free_version [] = {
    NULL,
    NULL,
    free_compute_tuile_opt,
    NULL,
    NULL,
    free_compute_tuile_opt,
    NULL,
    free_compute_tuile_opt,
    NULL,
    NULL,
    NULL
};

static inline void compute_case(int x, int y) {
    int n = (cur_img(y-1, x-1) !=0) +
            (cur_img(y-1, x)   !=0) +
            (cur_img(y-1, x+1) !=0) +
            (cur_img(y  , x-1) !=0) +
            (cur_img(y  , x+1) !=0) +
            (cur_img(y+1, x-1) !=0) +
            (cur_img(y+1, x)   !=0) +
            (cur_img(y+1, x+1) !=0);
    if (cur_img(y, x)) {
        if (n==2 || n==3)
            next_img(y,x) = couleur;
        else
            next_img(y,x) = 0;
    }
    else {
        if (n==3)
            next_img(y,x) = couleur;
        else
            next_img(y,x) = 0;
    }
}


///////////////////////////// Version séquentielle simple
unsigned compute_v0(unsigned nb_iter) {

    for (unsigned it = 1; it <= nb_iter; it ++) {

        for (int y = 1; y < DIM-1; y++)
            for (int x = 1; x < DIM-1; x++)
                compute_case(x,y);


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

#pragma omp parallel for schedule(static) collapse(2)
        for (int y = 1; y < DIM-1; y++)
            for (int x = 1; x < DIM-1; x++)
                compute_case(x,y);

        swap_images();
    }

    return 0;
}


/////////////////////////////
static inline void compute_tile(int tuilex, int tuiley) {
    for (int yloc = 0; yloc < GRAIN; yloc++) {
        for (int xloc = 0; xloc < GRAIN; xloc++) {
            unsigned y=tuiley*GRAIN+yloc;
            unsigned x=tuilex*GRAIN+xloc;
            if (x>0 && x<DIM-1 && y>0 && y<DIM-1)
                compute_case(x,y);
        }
    }
}

///////////////////////////// Version séquentielle avec tuiles
unsigned compute_v1(unsigned nb_iter){
    unsigned tranche = DIM / GRAIN;

    for (unsigned it = 1; it <= nb_iter; it ++) {

        for (unsigned tuiley = 0; tuiley < tranche; tuiley++)
            for (unsigned tuilex = 0; tuilex < tranche; tuilex++)

                compute_tile(tuilex, tuiley);

        swap_images();
    }

    return 0;
}


///////////////////////////// Version OpenMP avec tuiles
unsigned openMP_for_v1(unsigned nb_iter) {
    unsigned tranche = DIM / GRAIN;

    for (unsigned it = 1; it <= nb_iter; it ++) {

#pragma omp parallel for schedule(guided,4) collapse(2)
        for (unsigned tuiley = 0; tuiley < tranche; tuiley++)
            for (unsigned tuilex = 0; tuilex < tranche; tuilex++)
                compute_tile(tuilex, tuiley);

        swap_images();
    }

    return 0;
}


///////////////////////////// Version OpenMP avec tuiles et tâches
unsigned openMP_task_v1(unsigned nb_iter) {
    unsigned tranche = DIM / GRAIN;

    for (unsigned it = 1; it <= nb_iter; it ++) {

#pragma omp parallel
        {
#pragma omp single
            for (unsigned tuiley = 0; tuiley < tranche; tuiley++)
                for (unsigned tuilex = 0; tuilex < tranche; tuilex++)
#pragma omp task firstprivate(tuilex, tuiley)
                    compute_tile(tuilex, tuiley);

        }

        swap_images();
    }

    return 0;
}


///////////////////////////// Version optimisé fonctions générales
#define coord(y,x) (y+1)*tranche+x+1
bool* curr_unchanged = NULL;
bool* next_unchanged = NULL;
unsigned tranche;

static void swap_tiles() {
    bool* tmp = curr_unchanged;

    curr_unchanged = next_unchanged;
    next_unchanged = tmp;
}

void init_compute_tuile_opt() {
    tranche = DIM / GRAIN;

    //greater size in order to don't test if a tile is en a extreme position or not
    curr_unchanged = malloc(sizeof(unsigned)*(tranche+2)*(tranche+2));
    if (!curr_unchanged)
        exit(1);
    for(int tuilex=-1; tuilex<=tranche;tuilex++)
        for (int tuiley=-1; tuiley<=tranche; tuiley++)
            curr_unchanged[coord(tuilex,tuiley)] = false;

    next_unchanged = malloc(sizeof(unsigned)*(tranche+2)*(tranche+2));
    if (!next_unchanged)
        exit(1);
    for(int tuilex=-1; tuilex<=tranche;tuilex++)
        for (int tuiley=-1; tuiley<=tranche; tuiley++)
            next_unchanged[coord(tuilex,tuiley)] = false;
}

void free_compute_tuile_opt() {
    if (curr_unchanged) {
        free(curr_unchanged);
        curr_unchanged = NULL;
    }
    if (next_unchanged) {
        free(next_unchanged);
        next_unchanged = NULL;
    }
}


/////////////////////////////
static inline bool compute_tile_required(int tuilex, int tuiley) {
    return !(curr_unchanged[coord(tuiley,tuilex)] &&
            curr_unchanged[coord(tuiley+1,tuilex)] &&
            curr_unchanged[coord(tuiley-1,tuilex)] &&
            curr_unchanged[coord(tuiley,tuilex+1)] &&
            curr_unchanged[coord(tuiley,tuilex-1)] &&
            curr_unchanged[coord(tuiley+1,tuilex+1)] &&
            curr_unchanged[coord(tuiley+1,tuilex-1)] &&
            curr_unchanged[coord(tuiley-1,tuilex+1)] &&
            curr_unchanged[coord(tuiley-1,tuilex-1)]);
}

static inline bool compute_tile_changement(int tuilex, int tuiley) {
    bool tuile_unchanged = true;
    for (int yloc = 0; yloc < GRAIN; yloc++) {
        for (int xloc = 0; xloc < GRAIN; xloc++) {
            unsigned y=tuiley*GRAIN+yloc;
            unsigned x=tuilex*GRAIN+xloc;
            if (y>0 && y<DIM-1 && x>0 && x<DIM-1) {
                int n = (cur_img(y-1, x-1) !=0) +
                        (cur_img(y-1, x)   !=0) +
                        (cur_img(y-1, x+1) !=0) +
                        (cur_img(y  , x-1) !=0) +
                        (cur_img(y  , x+1) !=0) +
                        (cur_img(y+1, x-1) !=0) +
                        (cur_img(y+1, x)   !=0) +
                        (cur_img(y+1, x+1) !=0);
                if (cur_img(y, x)) {
                    if (n==2 || n==3)
                        next_img(y,x) = couleur;
                    else {
                        //cell dies
                        tuile_unchanged = false;
                        next_img(y,x) = 0;
                    }
                }
                else {
                    if (n==3) {
                        //cell creation
                        tuile_unchanged = false;
                        next_img(y,x) = couleur;
                    }
                    else
                        next_img(y,x) = 0;
                }
            }
        }
    }
    return tuile_unchanged;

}

static inline void compute_tile_opt(int tuilex, int tuiley) {
    if (compute_tile_required(tuilex, tuiley))
        next_unchanged[coord(tuiley,tuilex)] = compute_tile_changement(tuilex, tuiley);
    else
        next_unchanged[coord(tuiley,tuilex)] = true;
}


///////////////////////////// Version séquentielle optimisé
unsigned compute_v2(unsigned nb_iter)
{
    for (unsigned it = 1; it <= nb_iter; it ++) {

        for (int tuiley = 0; tuiley < tranche; tuiley++) {
            for (int tuilex = 0; tuilex < tranche; tuilex++) {
                compute_tile_opt(tuilex, tuiley);


            }
        }
        swap_tiles();
        swap_images();
    }

    return 0;
}


///////////////////////////// Version OpenMP for optimisée
unsigned openMP_for_v2(unsigned nb_iter)
{
    for (unsigned it = 1; it <= nb_iter; it ++) {

#pragma omp parallel for schedule(guided,4) collapse(2)
        for (int tuiley = 0; tuiley < tranche; tuiley++)
            for (int tuilex = 0; tuilex < tranche; tuilex++)
                compute_tile_opt(tuilex, tuiley);

        swap_tiles();
        swap_images();
    }

    return 0;

}


///////////////////////////// Version OpenMP task optimisé
unsigned openMP_task_v2(unsigned nb_iter) {

    for (unsigned it = 1; it <= nb_iter; it ++) {

#pragma omp parallel
        {
#pragma omp single
            for (int tuiley = 0; tuiley < tranche; tuiley++) {
                for (int tuilex = 0; tuilex < tranche; tuilex++) {
                    if (compute_tile_required(tuilex, tuiley)) {
#pragma omp task firstprivate(tuilex, tuiley)
                        next_unchanged[coord(tuiley,tuilex)] = compute_tile_changement(tuilex, tuiley);
                    }
                    else
                        next_unchanged[coord(tuiley,tuilex)] = true;
                }
            }
        }

        swap_tiles();
        swap_images();
    }

    return 0;
}


///////////////////////////// Version OpenCL

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v3 (unsigned nb_iter) {
    return ocl_compute (nb_iter);
}

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v4 (unsigned nb_iter) {
    return ocl_compute_opt (nb_iter);
}

#define GPU_FRAC 92

unsigned * picture;

// Renvoie le nombre d'itérations effectuées avant stabilisation, ou 0
unsigned compute_v5 (unsigned nb_iter) {
    static bool first = true;
    if (first) {
        first = false;
        picture = malloc(sizeof(unsigned) * DIM * DIM);
    }


    unsigned tranche = DIM / GRAIN;
    unsigned nb_tranches = compute_ratio(GPU_FRAC);
    unsigned cpu_calc = nb_tranches/GRAIN;
    //printf("%d\n", nb_tranches);

    for (unsigned it = 1; it <= nb_iter; it ++) {

        /*pthread_create(&thread, NULL, start, (void *) nb_iter);
        pthread_yield();*/
#pragma omp parallel
        {
#pragma omp single
            ocl_compute_hybrid(nb_iter, nb_tranches);
#pragma omp for schedule(guided,4) collapse(2)
            for (unsigned tuilex = cpu_calc-1; tuilex < tranche; tuilex++) {
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
                                    if (n==2 || n==3)
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
        }

        swap_images();
        get_picture_back(picture);
#pragma omp parallel for schedule(guided,4) collapse(2)
        for (int x=0; x<DIM; x++) {
            for (int y=0; y<nb_tranches; y++) {
                cur_img(y,x) = picture[y*DIM+x];
            }
        }
#pragma omp parallel for schedule(guided,4) collapse(2)
        for (int x=0; x<DIM; x++) {
            for (int y=nb_tranches; y<DIM; y++) {
                picture[y*DIM+x] = cur_img(y,x);
            }
        }
        put_picture(picture);

    }

    return 0;
}
