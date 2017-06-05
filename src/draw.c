#define _GNU_SOURCE

#include <SDL.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

#include "draw.h"
#include "graphics.h"

static unsigned couleur = 0xFFFF00FF; // Yellow

static void gun (int x, int y, int version)
{
    bool glider_gun [11][38] =
    {
        { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0 },
        { 0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0 },
        { 0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
        { 0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
        { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 },
    };

    if (version == 0)
        for (int i=0; i < 11; i++)
            for(int j=0; j < 38; j++)
                if (glider_gun [i][j])
                    cur_img (i+x, j+y) = couleur;

    if (version == 1)
        for (int i=0; i < 11; i++)
            for(int j=0; j < 38; j++)
                if (glider_gun [i][j])
                    cur_img (x-i, j+y) = couleur;

    if (version == 2)
        for (int i=0; i < 11; i++)
            for(int j=0; j < 38; j++)
                if (glider_gun [i][j])
                    cur_img (x-i, y-j) = couleur;

    if (version == 3)
        for (int i=0; i < 11; i++)
            for(int j=0; j < 38; j++)
                if (glider_gun [i][j])
                    cur_img (i+x, y-j) = couleur;

}

void draw_stable(void)
{
    for (int i=1; i < DIM-2; i+=4)
        for(int j=1; j < DIM-2; j+=4)
        {
            cur_img (i, j) =cur_img (i, (j+1)) =cur_img ((i+1), j) =cur_img ((i+1), (j+1)) = couleur;
        }
}

void draw_guns (void)
{
    memset(&cur_img(0,0), 0, DIM*DIM* sizeof(cur_img(0,0)));

    gun (0, 0, 0);
    gun (0,  DIM-1 , 3);
    gun (DIM - 1 , DIM - 1, 2);
    gun (DIM - 1 , 0, 1);

}

void draw_random (void)
{

    for (int i=1; i < DIM-1; i++)
        for(int j=1; j < DIM-1; j++)
        {
            cur_img (i, j) = random() & 01;
        }
}



static void spiral (int x, int y, int pas, int nbtours)
{
    int i = x, j = y, tour;

    for (tour = 1; tour <= nbtours; tour++) {
        for (; i < x + tour*pas;i++)
            cur_img (i,j) = couleur;
        for (; j < y + tour*pas+1;j++)
            cur_img (i,j) = couleur;
        for (; i > x - tour*pas-1 ;i--)
            cur_img (i,j) = couleur;
        for (; j > y - tour*pas-1;j--)
            cur_img (i,j) = couleur;
    }
}

void spiral_regular (int xdebut, int xfin, int ydebut, int yfin, int pas, int nbtours)
{
    int i,j;
    int taille = nbtours * pas + 2;

    for (i = xdebut + taille; i < xfin - taille; i += 2*taille)
        for (j = ydebut + taille; j < yfin - taille; j += 2*taille)
            spiral (i,j, pas, nbtours);
}


void draw_file(char *filename) {
    FILE *fp;
    char* line = NULL;
    size_t len = 0;
    ssize_t read;

    // open the file
    fp = fopen(filename, "r");
    if (fp==NULL) {
        printf("Unable to open the file\n");
        exit(EXIT_FAILURE);
    }

    //Init memory
    memset(&cur_img(0,0), 0, DIM*DIM* sizeof(cur_img(0,0)));

    bool coords = false;
    int x, y, xoffset, yoffset, currenty, currentx, currentNumber;
    currentNumber = 0;

    //parse the file
    while ((read = getline(&line, &len, fp)) != -1) {
        if (len>0 && line[0] != '#' && line[0] != '\n') {
            if (!coords) {
                sscanf(line, "x = %d, y = %d", &x, &y);
                printf("Pattern size : x = %d, y = %d\n", x, y);
                if (x>DIM) {
                    printf("Not enough columns\n");
                    exit(EXIT_FAILURE);
                }
                xoffset = (DIM - x)/2;
                currentx = xoffset;
                if (y>DIM) {
                    printf("Not enough rows\n");
                    exit(EXIT_FAILURE);
                }
                yoffset = (DIM - y)/2;
                currenty = yoffset;
                coords = true;
            }
            else {
                for (int n=0; n<strlen(line); n++) {
                    if (line[n] >= '0' && line[n] <= '9') {
                        //number, add it to the currentNumber counter
                        currentNumber = 10*currentNumber + line[n] - '0';
                    }
                    else if (line[n] == 'b' ||
                             line[n] == 'o' ||
                             line[n] == '$' ||
                             line[n] == '!'){
                        //printf("%d\n", currentNumber);
                        int repetitionNumber = (currentNumber==0) ? 1 : currentNumber;
                        if (line[n] == 'o') {
                            //printf n cells with the color
                            for (int i=0; i<repetitionNumber; i++)
                                cur_img (currenty,currentx+i) = couleur;
                        }
                        currentx+=repetitionNumber;
                        if (line[n] == '$') {
                            //end of line
                            currentx = xoffset;
                            currenty += repetitionNumber;
                        }
                        currentNumber = 0;

                        if (line[n] == '!')
                            return;

                    }
                }
            }

        }
    }
    if (line)
        free(line);
}
