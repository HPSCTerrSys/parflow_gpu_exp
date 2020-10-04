/*BHEADER*********************************************************************
 *
 *  Copyright (c) 1995-2009, Lawrence Livermore National Security,
 *  LLC. Produced at the Lawrence Livermore National Laboratory. Written
 *  by the Parflow Team (see the CONTRIBUTORS file)
 *  <parflow@lists.llnl.gov> CODE-OCEC-08-103. All rights reserved.
 *
 *  This file is part of Parflow. For details, see
 *  http://www.llnl.gov/casc/parflow
 *
 *  Please read the COPYRIGHT file or Our Notice and the LICENSE file
 *  for the GNU Lesser General Public License.
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License (as published
 *  by the Free Software Foundation) version 2.1 dated February 1999.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
 *  and conditions of the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
 *  USA
 **********************************************************************EHEADER*/

#include "amps.h"

static int COUNT_SEND = 10;
static int COUNT_WAIT = 10;

extern amps_GpuBuffer amps_gpu_recvbuf;
extern amps_GpuBuffer amps_gpu_sendbuf;

/* This CUDA stuff could be combined with AMPS_MPI_NOT_USE_PERSISTENT case */
#ifdef PARFLOW_HAVE_CUDA
void _amps_wait_exchange(amps_Handle handle)
{
  (void)handle;
}

void _amps_wait_exchange_internal(amps_Package package)
{
  COUNT_WAIT += 2;
  if(COUNT_SEND != COUNT_WAIT){ 
    printf("CALL COUNT PROBLEM: SEND: %d, WAIT: %d\n", COUNT_SEND,COUNT_WAIT);
    exit(1);
  }

  char *combuf;
  int errchk;
  int i;
  int size;
  MPI_Status *status;

  if (package->num_recv + package->num_send)
  {
    status = (MPI_Status*)calloc(2 * (package->num_recv +
                                  package->num_send), sizeof(MPI_Status));

    MPI_Waitall(2 * (package->num_recv + package->num_send),
                package->recv_requests, status);
    int bytes_corrupted = 0;
    for (i = 0; i < package->num_recv; i++)
    {
      int size_inv = amps_sizeof_invoice(amps_CommWorld, package->recv_invoices[i]);
      char *cpubuf = amps_gpu_recvbuf.buf_host[i];
      char *gpubuf = (char*)malloc(size_inv * sizeof(char));
      CUDA_ERRCHK(cudaMemcpy(gpubuf,
                    amps_gpu_recvbuf.buf[i], 
                      size_inv, cudaMemcpyDeviceToHost));
      for(int j = 0; j < size_inv; j++){
        if(cpubuf[j] != gpubuf[j]){
          bytes_corrupted++;
        }
      }
      if(bytes_corrupted != 0){
        printf("%d/%d bytes corrupted! (Rank: %d, COUNT: %d, Invoice: %d)\n", 
          bytes_corrupted, size_inv, amps_rank, COUNT_WAIT, i);
      }

      errchk = amps_gpupacking(AMPS_UNPACK, 
                 package->recv_invoices[i], 
                   i, &combuf, &size);
      if(errchk){
        printf("GPU unpacking failed at line: %d\n", errchk);
        exit(1);
      }
      if(size != size_inv){
        printf("size != size_inv! \n");
        exit(1);
      }
      free(gpubuf);
    }
    for (i = 0; i < package->num_recv; i++)
    {
      amps_gpu_sync_streams(i);
      AMPS_CLEAR_INVOICE(package->recv_invoices[i]);
    }
    free(status);
    (void)errchk;
  }

  for (i = 0; i < 2 * package->num_recv; i++)
  {
    if(package->recv_requests[i] != MPI_REQUEST_NULL)
      MPI_Request_free(&(package->recv_requests[i]));
  }
  for (i = 0; i < 2 * package->num_send; i++)
  {
    if(package->send_requests[i] != MPI_REQUEST_NULL)
      MPI_Request_free(&(package->send_requests[i]));
  }
}

amps_Handle amps_IExchangePackage(amps_Package package)
{
  char **combuf_cpu;
  int *size;
  int errchk;
  int i;

    /*--------------------------------------------------------------------
   * THIS IS DUMMY COMMUNICATION, SHOULD NOT INFLUENCE RESULTS
   *--------------------------------------------------------------------*/
  {
    int *sbuf = (int*)calloc(package->num_send, sizeof(int));
    int *rbuf = (int*)calloc(package->num_recv, sizeof(int));
    int num = package->num_send + package->num_recv;
    MPI_Request *rreq = (MPI_Request*)calloc((size_t)(num), sizeof(MPI_Request));
    MPI_Request *sreq = rreq + package->num_recv;
    MPI_Status *status = (MPI_Status*)calloc((size_t)(num), sizeof(MPI_Status));
    for (i = 0; i < package->num_send; i++){
      for (int j = 0; j < package->num_send; j++){
        if(i != j){
          if(package->dest[i] == package->dest[j]){
            printf("DUPLICATE DEST! num_send: %d\n",package->num_send);
            for (int k = 0; k < package->num_send; k++){
              printf("package->dest[%d]: %d\n",k,package->dest[k]);
            }
            exit(1);
          }
        }
      }
      sbuf[i] = 145;
      MPI_Isend(&sbuf[i], 1, MPI_INT, package->dest[i], 9, amps_CommWorld, &sreq[i]);
    }
    for (i = 0; i < package->num_recv; i++)
    {
      for (int j = 0; j < package->num_recv; j++){
        if(i != j){
          if(package->src[i] == package->src[j]){
            printf("DUPLICATE SRC! num_recv: %d\n",package->num_recv);
            for (int k = 0; k < package->num_recv; k++){
              printf("package->src[%d]: %d\n",k,package->src[k]);
            }
            exit(1);
          }
        }
      }
      MPI_Irecv(&rbuf[i], 1, MPI_INT, package->src[i], 9, amps_CommWorld, &rreq[i]);
    }
    MPI_Waitall(num, rreq, status);
    // MPI_Barrier(MPI_COMM_WORLD);
    for (i = 0; i < package->num_recv; i++)
    {
      if(rbuf[i] != 145){
        printf("ERROR in MPI Communication! \n");
        exit(1);
      }
    }
    free(sbuf);free(rbuf);free(rreq);free(status);
  }

  combuf_cpu = (char**)malloc(package->num_send * sizeof(char*));
  size = (int*)malloc(package->num_send * sizeof(int));

  /*--------------------------------------------------------------------
   * post receives for data to get
   *--------------------------------------------------------------------*/
  for (i = 0; i < package->num_recv; i++)
  {
    errchk = amps_gpupacking(AMPS_GETRBUF, package->recv_invoices[i], 
                   i, &combuf_cpu[0], &size[0]);
    if(errchk == 0){    
      package->recv_invoices[i]->mpi_type = MPI_BYTE;
    }
    else{ 
      printf("GPU recv packing check failed at line: %d\n", errchk);
      exit(1);
    }

    MPI_Irecv(combuf_cpu[0], size[0], package->recv_invoices[i]->mpi_type,
              package->src[i], COUNT_SEND, MPI_COMM_WORLD,
              &(package->recv_requests[i]));

    char *combuf_gpu = amps_gpu_recvbuf.buf[i];
    MPI_Irecv(combuf_gpu, size[0], package->recv_invoices[i]->mpi_type,
              package->src[i], COUNT_SEND + 1, MPI_COMM_WORLD,
              &(package->recv_requests[package->num_recv + i]));
  }

  /*--------------------------------------------------------------------
   * send out the data we have
   *--------------------------------------------------------------------*/
  for (i = 0; i < package->num_send; i++)
  {
    errchk = amps_gpupacking(AMPS_PACK, package->send_invoices[i], 
                   i, &combuf_cpu[i], &size[i]);
    if(errchk == 0){    
      package->send_invoices[i]->mpi_type = MPI_BYTE;
    }
    else{
      printf("GPU packing failed at line: %d\n", errchk);
      exit(1);
    }
  }
  for (i = 0; i < package->num_send; i++)
  {
    amps_gpu_sync_streams(i);
    
    char *combuf_gpu = amps_gpu_sendbuf.buf[i];
    //this copy is actually already done but here just for the clarity
    CUDA_ERRCHK(cudaMemcpy(combuf_gpu,
                  combuf_cpu[i], 
                    size[i], cudaMemcpyHostToDevice));

    MPI_Isend(combuf_cpu[i], size[i], package->send_invoices[i]->mpi_type,
              package->dest[i], COUNT_SEND, MPI_COMM_WORLD,
              &(package->send_requests[i]));

    MPI_Isend(combuf_gpu, size[i], package->send_invoices[i]->mpi_type,
              package->dest[i], COUNT_SEND + 1, MPI_COMM_WORLD,
              &(package->send_requests[package->num_send + i]));
  }
  free(combuf_cpu);
  free(size);

  // MPI_Status *status = (MPI_Status*)calloc(2 * (package->num_recv + package->num_send), sizeof(MPI_Status));
  // MPI_Waitall(2 * (package->num_recv + package->num_send), package->recv_requests, status);
  // MPI_Barrier(MPI_COMM_WORLD);
  // free(status);

  COUNT_SEND += 2;
  _amps_wait_exchange_internal(package);

  return(amps_NewHandle(amps_CommWorld, 0, NULL, package));
}

#elif AMPS_MPI_NOT_USE_PERSISTENT

void _amps_wait_exchange(amps_Handle handle)
{
  int notdone;
  int i;

  MPI_Status *status;

  if (handle->package->num_recv + handle->package->num_send)
  {
    status = (MPI_Status*)calloc((handle->package->num_recv +
                                  handle->package->num_send), sizeof(MPI_Status));

    MPI_Waitall(handle->package->num_recv + handle->package->num_send,
                handle->package->requests,
                status);

    free(status);

    for (i = 0; i < handle->package->num_recv; i++)
    {
      if (handle->package->recv_invoices[i]->mpi_type != MPI_DATATYPE_NULL)
      {
        MPI_Type_free(&(handle->package->recv_invoices[i]->mpi_type));
      }

      MPI_Request_free(&handle->package->requests[i]);
    }

    for (i = 0; i < handle->package->num_send; i++)
    {
      if (handle->package->send_invoices[i]->mpi_type != MPI_DATATYPE_NULL)
      {
        MPI_Type_free(&handle->package->send_invoices[i]->mpi_type);
      }

      MPI_Request_free(&handle->package->requests[handle->package->num_recv + i]);
    }
  }
}

amps_Handle amps_IExchangePackage(amps_Package package)
{
  int i;

  /*--------------------------------------------------------------------
   * post receives for data to get
   *--------------------------------------------------------------------*/
  package->recv_remaining = 0;

  for (i = 0; i < package->num_recv; i++)
  {
    amps_create_mpi_type(MPI_COMM_WORLD, package->recv_invoices[i]);

    MPI_Type_commit(&(package->recv_invoices[i]->mpi_type));

    MPI_Irecv(MPI_BOTTOM, 1, package->recv_invoices[i]->mpi_type,
              package->src[i], 0, MPI_COMM_WORLD,
              &(package->requests[i]));
  }

  /*--------------------------------------------------------------------
   * send out the data we have
   *--------------------------------------------------------------------*/
  for (i = 0; i < package->num_send; i++)
  {
    amps_create_mpi_type(MPI_COMM_WORLD, package->send_invoices[i]);

    MPI_Type_commit(&(package->send_invoices[i]->mpi_type));

    MPI_Isend(MPI_BOTTOM, 1, package->send_invoices[i]->mpi_type,
              package->dest[i], 0, MPI_COMM_WORLD,
              &(package->requests[package->num_recv + i]));
  }

  return(amps_NewHandle(amps_CommWorld, 0, NULL, package));
}

#else

void _amps_wait_exchange(amps_Handle handle)
{
  int i;
  int num;

  num = handle->package->num_send + handle->package->num_recv;

  if (num)
  {
    if (handle->package->num_recv)
    {
      for (i = 0; i < handle->package->num_recv; i++)
      {
        AMPS_CLEAR_INVOICE(handle->package->recv_invoices[i]);
      }
    }

    MPI_Waitall(num, handle->package->recv_requests,
                handle->package->status);
  }

#ifdef AMPS_MPI_PACKAGE_LOWSTORAGE
  /* Needed by the DEC's; need better memory allocation strategy */
  /* Need to uncommit packages when not in use */
  /* amps_Commit followed by amps_UnCommit ????? */
  if (handle->package->commited)
  {
    for (i = 0; i < handle->package->num_recv; i++)
    {
      if (handle->package->recv_invoices[i]->mpi_type != MPI_DATATYPE_NULL)
      {
        MPI_Type_free(&(handle->package->recv_invoices[i]->mpi_type));
      }

      MPI_Request_free(&(handle->package->recv_requests[i]));
    }

    for (i = 0; i < handle->package->num_send; i++)
    {
      if (handle->package->send_invoices[i]->mpi_type != MPI_DATATYPE_NULL)
      {
        MPI_Type_free(&handle->package->send_invoices[i]->mpi_type);
      }

      MPI_Request_free(&(handle->package->send_requests[i]));
    }

    if (handle->package->recv_requests)
    {
      free(handle->package->recv_requests);
      handle->package->recv_requests = NULL;
    }
    if (handle->package->status)
    {
      free(handle->package->status);
      handle->package->status = NULL;
    }

    handle->package->commited = FALSE;
  }
#endif
}

/*===========================================================================*/
/**
 *
 * The \Ref{amps_IExchangePackage} initiates the communication of the
 * invoices found in the {\bf package} structure that is passed in.  Once a
 * \Ref{amps_IExchangePackage} is issued it is illegal to access the
 * variables that are being communicated.  An \Ref{amps_IExchangePackage}
 * is always followed by an \Ref{amps_Wait} on the {\bf handle} that is
 * returned.
 *
 * {\large Example:}
 * \begin{verbatim}
 * // Initialize exchange of boundary points
 * handle = amps_IExchangePackage(package);
 *
 * // Compute on the "interior points"
 *
 * // Wait for the exchange to complete
 * amps_Wait(handle);
 * \end{verbatim}
 *
 * {\large Notes:}
 *
 * This routine can be optimized on some architectures so if your
 * communication can be formulated using it there might be
 * some performance advantages.
 *
 * @memo Initiate package communication
 * @param package the collection of invoices to communicate
 * @return Handle for the asynchronous communication
 */
amps_Handle amps_IExchangePackage(amps_Package package)
{
  int i;
  int num;

  num = package->num_send + package->num_recv;

  /*-------------------------------------------------------------------
  * Check if we need to allocate the MPI types and requests
  *------------------------------------------------------------------*/
  if (!package->commited)
  {
    package->commited = TRUE;

    /*--------------------------------------------------------------------
     * Allocate the arrays need for MPI
     *--------------------------------------------------------------------*/
    if (num)
    {
      package->recv_requests = (MPI_Request*)calloc((size_t)(num),
                                                    sizeof(MPI_Request));

      package->status = (MPI_Status*)calloc((size_t)(num),
                                            sizeof(MPI_Status));

      package->send_requests = package->recv_requests +
                               package->num_recv;
    }

    /*--------------------------------------------------------------------
     * Set up the receive types and requests
     *--------------------------------------------------------------------*/
    if (package->num_recv)
    {
      for (i = 0; i < package->num_recv; i++)
      {
        amps_create_mpi_type(MPI_COMM_WORLD, package->recv_invoices[i]);
        MPI_Type_commit(&(package->recv_invoices[i]->mpi_type));

        // Temporaries needed by insure++
        MPI_Datatype type = package->recv_invoices[i]->mpi_type;
        MPI_Request *request_ptr = &(package->recv_requests[i]);
        MPI_Recv_init(MPI_BOTTOM, 1,
                      type,
                      package->src[i], 0, MPI_COMM_WORLD,
                      request_ptr);
      }
    }

    /*--------------------------------------------------------------------
     * Set up the send types and requests
     *--------------------------------------------------------------------*/
    if (package->num_send)
    {
      for (i = 0; i < package->num_send; i++)
      {
        amps_create_mpi_type(MPI_COMM_WORLD,
                             package->send_invoices[i]);

        MPI_Type_commit(&(package->send_invoices[i]->mpi_type));

        // Temporaries needed by insure++
        MPI_Datatype type = package->send_invoices[i]->mpi_type;
        MPI_Request* request_ptr = &(package->send_requests[i]);
        MPI_Ssend_init(MPI_BOTTOM, 1,
                       type,
                       package->dest[i], 0, MPI_COMM_WORLD,
                       request_ptr);
      }
    }
  }

  if (num)
  {
    /*--------------------------------------------------------------------
     * post send and receives
     *--------------------------------------------------------------------*/
    MPI_Startall(num, package->recv_requests);
  }


  return(amps_NewHandle(amps_CommWorld, 0, NULL, package));
}

#endif

