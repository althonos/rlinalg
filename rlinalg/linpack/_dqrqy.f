c
      subroutine dqrqy(x, ldx, p, n, k, qraux, y, ny)

      integer ldx, n, k, ny, p
      double precision x(ldx, p), qraux(p), y(n,ny)
      integer info, j
      double precision dummy(1)
      do j = 1,ny
          call dqrsl(x, ldx, n, k, qraux, y(1,j), y(1,j), dummy,
     &               dummy, dummy, dummy, 10000, info)
      end do
      return
      end
