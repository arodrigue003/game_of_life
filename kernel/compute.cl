__kernel void transpose_naif (__global unsigned *in, __global unsigned *out)
{
  int x = get_global_id (0);
  int y = get_global_id (1);

  out [x * DIM + y] = in [y * DIM + x];
}



__kernel void transpose (__global unsigned *in, __global unsigned *out)
{
  __local unsigned tile [TILEX][TILEY+1];
  int x = get_global_id (0);
  int y = get_global_id (1);
  int xloc = get_local_id (0);
  int yloc = get_local_id (1);

  tile [xloc][yloc] = in [y * DIM + x];

  barrier (CLK_LOCAL_MEM_FENCE);

  out [(x - xloc + yloc) * DIM + y - yloc + xloc] = tile [yloc][xloc];
}



// NE PAS MODIFIER
static unsigned color_mean (unsigned c1, unsigned c2)
{
  uchar4 c;

  c.x = ((unsigned)(((uchar4 *) &c1)->x) + (unsigned)(((uchar4 *) &c2)->x)) / 2;
  c.y = ((unsigned)(((uchar4 *) &c1)->y) + (unsigned)(((uchar4 *) &c2)->y)) / 2;
  c.z = ((unsigned)(((uchar4 *) &c1)->z) + (unsigned)(((uchar4 *) &c2)->z)) / 2;
  c.w = ((unsigned)(((uchar4 *) &c1)->w) + (unsigned)(((uchar4 *) &c2)->w)) / 2;

  return (unsigned) c;
}

// NE PAS MODIFIER
static int4 color_to_int4 (unsigned c)
{
  uchar4 ci = *(uchar4 *) &c;
  return convert_int4 (ci);
}

// NE PAS MODIFIER
static unsigned int4_to_color (int4 i)
{
  return (unsigned) convert_uchar4 (i);
}



// NE PAS MODIFIER
static float4 color_scatter (unsigned c)
{
  uchar4 ci;

  ci.s0123 = (*((uchar4 *) &c)).s3210;
  return convert_float4 (ci) / (float4) 255;
}

// NE PAS MODIFIER: ce noyau est appelé lorsqu'une mise à jour de la
// texture de l'image affichée est requise
__kernel void update_texture (__global unsigned *cur, __write_only image2d_t tex)
{
  int y = get_global_id (1);
  int x = get_global_id (0);
  int2 pos = (int2)(x, y);
  unsigned c;

  c = cur [y * DIM + x];

  write_imagef (tex, pos, color_scatter (c));
}

__kernel void compute_pixel_naif(__global unsigned *in, __global unsigned *out) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x>0 && x<DIM-1 && y>0 && y<DIM-1) {
    int n = (in[(y-1)*DIM+x-1] !=0) +
            (in[(y)*DIM+x-1]   !=0) +
            (in[(y+1)*DIM+x-1] !=0) +
            (in[(y-1)*DIM+x  ] !=0) +
            (in[(y+1)*DIM+x  ] !=0) +
            (in[(y-1)*DIM+x+1] !=0) +
            (in[(y)*DIM+x+1]   !=0) +
            (in[(y+1)*DIM+x+1] !=0);
        out[y*DIM+x] = (n==3 || (n==2 && in[(y)*DIM+x])) * 0xFFFF00FF;
    }
}

__kernel void compute_pixel_tuile(__global unsigned *in, __global unsigned *out) {
    __local unsigned tile[TILEY+2][TILEX+2];
    int x = get_global_id(0);
    int y = get_global_id(1);
    int xloc = get_local_id(0);
    int yloc = get_local_id(1);
    tile[yloc+1][xloc+1] = in[y * DIM + x]; //The local size area is in the center of the tile
    //load values upside to the area
    if ((y>0) && (yloc == 0)) {
        tile[0][xloc+1] = in[(y-1)*DIM + x];
        if (xloc==0)
            tile[0][0] = in[(y-1)*DIM+x-1];
        if (xloc==TILEX-1)
            tile[0][TILEX+1] = in[(y-1)*DIM+x+1];
    }
    //load values below the area
    if ((y<DIM-1) && (yloc == TILEY-1)) {
      tile[yloc+1+1][xloc+1] = in[(y+1)*DIM + x];
      if (yloc==0)
          tile[TILEY+1][0] = in[(y+1)*DIM+x-1];
      if (yloc==TILEY-1)
          tile[TILEY+1][TILEX+1] = in[(y+1)*DIM+x+1];
    }
    //load values left to the area
    if ((x>0) && (xloc == 0))
      tile[yloc+1][xloc+1-1] = in[y*DIM + x-1];
    //load values right to the area
    if ((x<DIM-1) && (xloc == TILEX-1))
      tile[yloc+1][xloc+1+1] = in[y*DIM + x+1];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (x>0 && y>0 && x<DIM-1 && y<DIM-1) {
    int n = (tile[yloc-1+1][xloc-1+1] !=0) +
            (tile[yloc-1+1][xloc+1]   !=0) +
            (tile[yloc-1+1][xloc+1+1] !=0) +
            (tile[yloc+1][xloc-1+1] !=0) +
            (tile[yloc+1][xloc+1+1] !=0) +
            (tile[yloc+1+1][xloc-1+1] !=0) +
            (tile[yloc+1+1][xloc+1]   !=0) +
            (tile[yloc+1+1][xloc+1+1] !=0);
        out[y*DIM+x] = (n==3 || (n==2 && tile[yloc+1][xloc+1])) * 0xFFFF00FF;
    }
}

__kernel void compute_pixel_opt(__global unsigned *in, __global unsigned *out, __global bool *curr_unchanged, __global bool *next_unchanged) {
    __local unsigned tile[TILEY+2][TILEX+2];
    int x = get_global_id(0);
    int y = get_global_id(1);
    int xloc = get_local_id(0);
    int yloc = get_local_id(1);
    tile[yloc+1][xloc+1] = in[y * DIM + x]; //The local size area is in the center of the tile
    //load values upside to the area
    if ((y>0) && (yloc == 0)) {
        tile[yloc+1-1][xloc+1] = in[(y-1)*DIM + x];
        if (xloc==0)
            tile[0][0] = in[(y-1)*DIM+x-1];
        if (xloc==TILEX-1)
            tile[0][TILEX+1] = in[(y-1)*DIM+x+1];
    }
    //load values below the area
    if ((y<DIM-1) && (yloc == TILEY-1)) {
      tile[yloc+1+1][xloc+1] = in[(y+1)*DIM + x];
      if (yloc==0)
          tile[TILEY+1][0] = in[(y+1)*DIM+x-1];
      if (yloc==TILEY-1)
          tile[TILEY+1][TILEX+1] = in[(y+1)*DIM+x+1];
    }
    //load values left to the area
    if ((x>0) && (xloc == 0))
      tile[yloc+1][xloc+1-1] = in[y*DIM + x-1];
    //load values right to the area
    if ((x<DIM-1) && (xloc == TILEX-1))
      tile[yloc+1][xloc+1+1] = in[y*DIM + x+1];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (x>0 && y>0 && x<DIM-1 && y<DIM-1) {
    int n = (tile[yloc-1+1][xloc-1+1] !=0) +
            (tile[yloc-1+1][xloc+1]   !=0) +
            (tile[yloc-1+1][xloc+1+1] !=0) +
            (tile[yloc+1][xloc-1+1] !=0) +
            (tile[yloc+1][xloc+1+1] !=0) +
            (tile[yloc+1+1][xloc-1+1] !=0) +
            (tile[yloc+1+1][xloc+1]   !=0) +
            (tile[yloc+1+1][xloc+1+1] !=0);
        out[y*DIM+x] = (n==3 || (n==2 && tile[yloc+1][xloc+1])) * 0xFFFF00FF;
    }
}
