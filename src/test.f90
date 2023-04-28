program test
    use mpi
    use cudafor
    use mod_nvtx
    use openacc

    implicit none
    ! MPI vars
    CHARACTER(len=10) :: processor_name
    integer           :: ierr_mpi, myRank, nRanks, localRank, peer

    ! CUDA and OpenACC vars
    type(cudaDeviceProp) :: deviceProp
    integer              :: ierr_cuda, deviceCount

    ! Actual variables
    real(4), allocatable :: chunk_recv(:), chunk_send(:)
    integer              :: arr_size, chunk_size, i, iter, numIters

    ! Bind device to local rank
    call GET_ENVIRONMENT_VARIABLE('OMPI_COMM_WORLD_LOCAL_RANK', processor_name)
    read(processor_name,'(i10)') localRank
    ierr_cuda = cudaSetDevice(localRank)

    ! Initialize MPI env
    call MPI_Init(ierr_mpi)

    ! Get MPI rank and number of ranks
    call MPI_Comm_rank(MPI_COMM_WORLD, myRank, ierr_mpi)
    call MPI_Comm_size(MPI_COMM_WORLD, nRanks, ierr_mpi)

    ! Get total number of devices across nodes
    ierr_cuda = cudaGetDeviceCount(deviceCount)

    ! Every rank gets device properties
    ierr_cuda = cudaGetDeviceProperties(deviceProp, localRank)

    ! Set array size, and check that its divisible by number of ranks
    arr_size = 16000000
    if (mod(arr_size, nRanks) .ne. 0) then
        if (myRank .eq. 0) write(*,*) 'ERROR: Array size must be divisible by number of ranks'
        stop
    end if

    ! Set chunk size and allocate arrays
    chunk_size = arr_size / nRanks
    allocate(chunk_recv(chunk_size))
    allocate(chunk_send(chunk_size))

    ! Create chunks on GPU memory, transfer other data to GPU
    !$acc enter data create(chunk_recv(:), chunk_send(:)) copyin(myRank, chunk_size)

    ! Initialize chunk_recv to zero
    !$acc kernels present(chunk_recv(:))
    chunk_recv(:) = 0.0
    !$acc end kernels

    ! Initialize chunk_send for rank 0
    if (myRank .eq. 0) then
        !$acc parallel loop gang vector present(chunk_send(:), chunk_recv(:), chunk_size) private(i)
        do i = 1,chunk_size
            chunk_send(i) = real(i,4)
            chunk_recv(i) = chunk_send(i)
        end do
        !$acc end parallel loop
    end if

    ! Barrier before proceeding
    call MPI_Barrier(MPI_COMM_WORLD, ierr_mpi)

    do iter = 1,10
        ! Rank zero send the message to all
        if (myRank .eq. 0) then
            do peer = 1,nRanks-1
                !$acc host_data use_device(chunk_send(:))
                call MPI_Send(chunk_send, chunk_size, MPI_REAL, peer, 0, MPI_COMM_WORLD, ierr_mpi)
                !$acc end host_data
            end do
        else
            !$acc host_data use_device(chunk_recv(:))
            call MPI_Recv(chunk_recv, chunk_size, MPI_REAL, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr_mpi)
            !$acc end host_data
        end if
        !$acc parallel loop gang vector present(chunk_recv(:), myRank)
        do i = 1,chunk_size
            chunk_recv(i) = chunk_recv(i) + myRank
        end do
        !$acc end parallel loop
    end do

    ! Barrier before proceeding
    call MPI_Barrier(MPI_COMM_WORLD, ierr_mpi)

    ! Copy chunk_recv back to host and destroy chunk_send
    !$acc exit data copyout(chunk_recv(:)) delete(chunk_send(:))

    ! Every process prints its chunk
#ifdef DEBUG
    do i = 1,chunk_size
        write(*,*) 'Rank ', myRank, ' chunk_recv(', i, ') = ', chunk_recv(i)
    end do
#endif

    ! Finalize MPI env
    call MPI_Finalize(ierr_mpi)
end program test