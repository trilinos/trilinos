// @HEADER
// *****************************************************************************
//       Ifpack2: Templated Object-Oriented Algebraic Preconditioner Package
//
// Copyright 2009 NTESS and the Ifpack2 contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

#ifndef IFPACK2_BLOCKCOMPUTERES_IMPL_HPP
#define IFPACK2_BLOCKCOMPUTERES_IMPL_HPP

#include "Ifpack2_BlockHelper.hpp"

namespace Ifpack2 {

  namespace BlockHelperDetails {

    ///
    /// A - Tridiags(A), i.e., R in the splitting A = D + R.
    ///
    template <typename MatrixType>
    struct AmD {
      using impl_type = BlockHelperDetails::ImplType<MatrixType>;
      using local_ordinal_type_1d_view = typename impl_type::local_ordinal_type_1d_view;
      using size_type_1d_view = typename impl_type::size_type_1d_view;
      using i64_2d_view  = typename impl_type::i64_2d_view;
      using impl_scalar_type_1d_view_tpetra = Unmanaged<typename impl_type::impl_scalar_type_1d_view_tpetra>;
      // rowptr points to the start of each row of A_colindsub.
      size_type_1d_view rowptr, rowptr_remote;
      // Indices into A's rows giving the blocks to extract. rowptr(i) points to
      // the i'th row. Thus, g.entries(A_colindsub(rowptr(row) : rowptr(row+1))),
      // where g is A's graph, are the columns AmD uses. If seq_method_, then
      // A_colindsub contains all the LIDs and A_colindsub_remote is empty. If !
      // seq_method_, then A_colindsub contains owned LIDs and A_colindsub_remote
      // contains the remote ones.
      local_ordinal_type_1d_view A_colindsub, A_colindsub_remote;
      // Precomputed direct offsets to A,x blocks, for owned entries (OverlapTag case) or all entries (AsyncTag case)
      i64_2d_view A_offsets;
      i64_2d_view x_offsets;
      // Precomputed direct offsets to A,x blocks, for non-owned entries (OverlapTag case). For AsyncTag case this is left empty.
      i64_2d_view A_offsets_remote;
      i64_2d_view x_offsets_remote;

      // Currently always true.
      bool is_tpetra_block_crs;

      // If is_tpetra_block_crs, then this is a pointer to A_'s value data.
      impl_scalar_type_1d_view_tpetra tpetra_values;

      AmD() = default;
      AmD(const AmD &b) = default;
    };

    template<typename MatrixType>
    struct PartInterface {
      using local_ordinal_type = typename BlockHelperDetails::ImplType<MatrixType>::local_ordinal_type;
      using local_ordinal_type_1d_view = typename BlockHelperDetails::ImplType<MatrixType>::local_ordinal_type_1d_view;
      using local_ordinal_type_2d_view = typename BlockHelperDetails::ImplType<MatrixType>::local_ordinal_type_2d_view;

      PartInterface() = default;
      PartInterface(const PartInterface &b) = default;

      // Some terms:
      //   The matrix A is split as A = D + R, where D is the matrix of tridiag
      // blocks and R is the remainder.
      //   A part is roughly a synonym for a tridiag. The distinction is that a part
      // is the set of rows belonging to one tridiag and, equivalently, the off-diag
      // rows in R associated with that tridiag. In contrast, the term tridiag is
      // used to refer specifically to tridiag data, such as the pointer into the
      // tridiag data array.
      //   Local (lcl) row are the LIDs. lclrow lists the LIDs belonging to each
      // tridiag, and partptr points to the beginning of each tridiag. This is the
      // LID space.
      //   Row index (idx) is the ordinal in the tridiag ordering. lclrow is indexed
      // by this ordinal. This is the 'index' space.
      //   A flat index is the mathematical index into an array. A pack index
      // accounts for SIMD packing.

      // Local row LIDs. Permutation from caller's index space to tridiag index
      // space.
      local_ordinal_type_1d_view lclrow;
      // partptr_ is the pointer array into lclrow_.
      local_ordinal_type_1d_view partptr; // np+1
      local_ordinal_type_2d_view partptr_sub;
      local_ordinal_type_1d_view partptr_schur;
      // packptr_(i), for i the pack index, indexes partptr_. partptr_(packptr_(i))
      // is the start of the i'th pack.
      local_ordinal_type_1d_view packptr; // npack+1
      local_ordinal_type_1d_view packptr_sub;
      local_ordinal_type_1d_view packindices_sub;
      local_ordinal_type_2d_view packindices_schur;
      // part2rowidx0_(i) is the flat row index of the start of the i'th part. It's
      // an alias of partptr_ in the case of no overlap.
      local_ordinal_type_1d_view part2rowidx0; // np+1
      local_ordinal_type_1d_view part2rowidx0_sub;
      // part2packrowidx0_(i) is the packed row index. If vector_length is 1, then
      // it's the same as part2rowidx0_; if it's > 1, then the value is combined
      // with i % vector_length to get the location in the packed data.
      local_ordinal_type_1d_view part2packrowidx0; // np+1
      local_ordinal_type_2d_view part2packrowidx0_sub;
      local_ordinal_type part2packrowidx0_back; // So we don't need to grab the array from the GPU.
      // rowidx2part_ maps the row index to the part index.
      local_ordinal_type_1d_view rowidx2part; // nr
      local_ordinal_type_1d_view rowidx2part_sub;
      // True if lcl{row|col} is at most a constant away from row{idx|col}. In
      // practice, this knowledge is not particularly useful, as packing for batched
      // processing is done at the same time as the permutation from LID to index
      // space. But it's easy to detect, so it's recorded in case an optimization
      // can be made based on it.
      bool row_contiguous;

      local_ordinal_type max_partsz;
      local_ordinal_type max_subpartsz;
      local_ordinal_type n_subparts_per_part;
      local_ordinal_type nparts;
    };

    // Precompute offsets of each A and x entry to speed up residual.
    // (Applies for hasBlockCrsMatrix == true and OverlapTag/AsyncTag)
    // Reading A, x take up to 4, 6 levels of indirection respectively,
    // but precomputing the offsets reduces it to 2 for both.
    //
    // This function allocates and populates these members of AmD:
    // A_offsets, x_offsets, A_offsets_remote, x_offsets_remote
    template<typename MatrixType>
    void precompute_Ax_offsets(
      AmD<MatrixType> &amd,
      const PartInterface<MatrixType> &interf,
      const Teuchos::RCP<const typename ImplType<MatrixType>::tpetra_crs_graph_type> &g,
      const typename ImplType<MatrixType>::local_ordinal_type_1d_view &dm2cm,
      int blocksize,
      bool ownedRemoteSeparate)
    {
        using impl_type = ImplType<MatrixType>;
        using i64_2d_view = typename impl_type::i64_2d_view;
        using size_type = typename impl_type::size_type;
        using local_ordinal_type = typename impl_type::local_ordinal_type;
        using execution_space = typename impl_type::execution_space;
        std::cout << "Hello from symbolic phase: precompuing offsets\n";
        auto local_graph = g->getLocalGraphDevice();
        const auto A_block_rowptr = local_graph.row_map;
        const auto A_colind = local_graph.entries;
        local_ordinal_type numLocalRows = interf.rowidx2part.extent(0);
        int blocksize_square = blocksize * blocksize;
        // shallow-copying views to avoid capturing the amd, interf objects in lambdas
        auto lclrow = interf.lclrow;
        auto A_colindsub = amd.A_colindsub;
        auto A_colindsub_remote = amd.A_colindsub_remote;
        auto rowptr = amd.rowptr;
        auto rowptr_remote = amd.rowptr_remote;
        bool is_dm2cm_active = dm2cm.extent(0);
        std::cout << "ownedRemoteSeparate ? " << ownedRemoteSeparate << '\n';
        std::cout << "is_dm2cm_active? " << is_dm2cm_active << '\n';
        std::cout << "blocksize = " << blocksize << '\n';
        std::cout << "numLocalRows = " << numLocalRows << '\n';
        if(ownedRemoteSeparate) {
          std::cout << "In overlap case - separate owned and remote\n";
          // amd.rowptr points to owned entries only, and amd.rowptr_remote points to nonowned.
          local_ordinal_type maxOwnedEntriesPerRow;
          local_ordinal_type maxNonownedEntriesPerRow;
          std::cout << "1\n";
          Kokkos::parallel_reduce(Kokkos::RangePolicy<execution_space>(0, numLocalRows),
            KOKKOS_LAMBDA(local_ordinal_type i, local_ordinal_type& lmaxOwned, local_ordinal_type& lmaxNonowned)
            {
              const local_ordinal_type lr = lclrow(i);
              local_ordinal_type rowNumOwned = rowptr(lr+1) - rowptr(lr);
              local_ordinal_type rowNumNonowned = rowptr_remote(lr+1) - rowptr_remote(lr);
              if(rowNumOwned > lmaxOwned)
                lmaxOwned = rowNumOwned;
              if(rowNumNonowned > lmaxNonowned)
                lmaxNonowned = rowNumNonowned;
            }, Kokkos::Max<local_ordinal_type>(maxOwnedEntriesPerRow), Kokkos::Max<local_ordinal_type>(maxNonownedEntriesPerRow));
          std::cout << "2\n";
          // Allocate the four offsets views now that we know the dimensions
          amd.A_offsets = i64_2d_view("amd.A_offsets", numLocalRows, maxOwnedEntriesPerRow);
          amd.x_offsets = i64_2d_view("amd.x_offsets", numLocalRows, maxOwnedEntriesPerRow);
          amd.A_offsets_remote = i64_2d_view("amd.A_offsets_remote", numLocalRows, maxNonownedEntriesPerRow);
          amd.x_offsets_remote = i64_2d_view("amd.x_offsets_remote", numLocalRows, maxNonownedEntriesPerRow);
          auto A_offsets = amd.A_offsets;
          auto x_offsets = amd.x_offsets;
          auto A_offsets_remote = amd.A_offsets_remote;
          auto x_offsets_remote = amd.x_offsets_remote;
          // Now, populate all the offsets. Use ArithTraits<int64_t>::min to say that no entry is not present.
          Kokkos::parallel_for(Kokkos::RangePolicy<execution_space>(0, numLocalRows),
            KOKKOS_LAMBDA(local_ordinal_type i)
            {
              const local_ordinal_type lr = lclrow(i);
              const size_type A_k0 = A_block_rowptr(lr);
              // Owned entries
              size_type rowBegin = rowptr(lr);
              local_ordinal_type rowNumOwned = rowptr(lr+1) - rowBegin;
              for(local_ordinal_type entry = 0; entry < maxOwnedEntriesPerRow; entry++) {
                if(entry < rowNumOwned) {
                  const size_type j = A_k0 + A_colindsub(rowBegin + entry);
                  const local_ordinal_type A_colind_at_j = A_colind(j);
                  const auto loc = is_dm2cm_active ? dm2cm(A_colind_at_j) : A_colind_at_j;
                  A_offsets(i, entry) = j * blocksize_square;
                  x_offsets(i, entry) = loc * blocksize;
                }
                else {
                  A_offsets(i, entry) = Kokkos::ArithTraits<int64_t>::min();
                  x_offsets(i, entry) = Kokkos::ArithTraits<int64_t>::min();
                }
              }
              // Nonowned entries
              rowBegin = rowptr_remote(lr);
              local_ordinal_type rowNumNonowned = rowptr_remote(lr+1) - rowBegin;
              for(local_ordinal_type entry = 0; entry < maxNonownedEntriesPerRow; entry++) {
                if(entry < rowNumNonowned) {
                  const size_type j = A_k0 + A_colindsub_remote(rowBegin + entry);
                  const local_ordinal_type A_colind_at_j = A_colind(j);
                  const auto loc = A_colind_at_j - numLocalRows;
                  A_offsets_remote(i, entry) = j * blocksize_square;
                  x_offsets_remote(i, entry) = loc * blocksize;
                }
                else {
                  A_offsets_remote(i, entry) = Kokkos::ArithTraits<int64_t>::min();
                  x_offsets_remote(i, entry) = Kokkos::ArithTraits<int64_t>::min();
                }
              }
            });
          Kokkos::fence();
          std::cout << "3\n";
        }
        else {
          std::cout << "In non-overlap case - owned and remote in same rowptrs\n";
          // amd.rowptr points to both owned and nonowned entries, so it tells us how many columns A_offsets,x_offsets should have.
          local_ordinal_type maxEntriesPerRow;
          std::cout << "4\n";
          Kokkos::parallel_reduce(Kokkos::RangePolicy<execution_space>(0, numLocalRows),
            KOKKOS_LAMBDA(local_ordinal_type i, local_ordinal_type& lmax)
            {
              const local_ordinal_type lr = lclrow(i);
              local_ordinal_type rowNum = rowptr(lr+1) - rowptr(lr);
              if(rowNum > lmax)
                lmax = rowNum;
            }, Kokkos::Max<local_ordinal_type>(maxEntriesPerRow));
          Kokkos::fence();
          std::cout << "4.5\n";
          amd.A_offsets = i64_2d_view("amd.A_offsets", numLocalRows, maxEntriesPerRow);
          amd.x_offsets = i64_2d_view("amd.x_offsets", numLocalRows, maxEntriesPerRow);
          auto A_offsets = amd.A_offsets;
          auto x_offsets = amd.x_offsets;
          std::cout << "5\n";
          //Populate A,x offsets. Use ArithTraits<int64_t>::min to say that no entry is not present.
          //For x offsets, add a shift blocksize*numLocalRows to represent that it indexes into x_remote instead of x.
          Kokkos::parallel_for(Kokkos::RangePolicy<execution_space>(0, numLocalRows),
            KOKKOS_LAMBDA(local_ordinal_type i)
            {
              const local_ordinal_type lr = lclrow(i);
              const size_type A_k0 = A_block_rowptr(lr);
              // Owned entries
              size_type rowBegin = rowptr(lr);
              local_ordinal_type rowOwned = rowptr(lr+1) - rowBegin;
              for(local_ordinal_type entry = 0; entry < maxEntriesPerRow; entry++) {
                if(entry < rowOwned) {
                  const size_type j = A_k0 + A_colindsub(rowBegin + entry);
                  A_offsets(i, entry) = j * blocksize_square;
                  const local_ordinal_type A_colind_at_j = A_colind(j);
                  if(A_colind_at_j < numLocalRows) {
                    const auto loc = is_dm2cm_active ? dm2cm[A_colind_at_j] : A_colind_at_j;
                    x_offsets(i, entry) = loc * blocksize;
                  }
                  else {
                    x_offsets(i, entry) = A_colind_at_j * blocksize;
                  }
                }
                else {
                  A_offsets(i, entry) = Kokkos::ArithTraits<int64_t>::min();
                  x_offsets(i, entry) = Kokkos::ArithTraits<int64_t>::min();
                }
              }
            });
          Kokkos::fence();
          std::cout << "6\n";
        }
    }

    ///
    /// compute local residula vector y = b - R x
    ///
    static inline int ComputeResidualVectorRecommendedCudaVectorSize(const int blksize,
                                                                     const int team_size) {
      int total_team_size(0);
      if      (blksize <=  5) total_team_size =  32;
      else if (blksize <=  9) total_team_size =  32; // 64
      else if (blksize <= 12) total_team_size =  96;
      else if (blksize <= 16) total_team_size = 128;
      else if (blksize <= 20) total_team_size = 160;
      else                    total_team_size = 160;
      return total_team_size/team_size;
    }

    static inline int ComputeResidualVectorRecommendedHIPVectorSize(const int blksize,
								    const int team_size) {
      int total_team_size(0);
      if      (blksize <=  5) total_team_size =  32;
      else if (blksize <=  9) total_team_size =  32; // 64
      else if (blksize <= 12) total_team_size =  96;
      else if (blksize <= 16) total_team_size = 128;
      else if (blksize <= 20) total_team_size = 160;
      else                    total_team_size = 160;
      return total_team_size/team_size;
    }

    static inline int ComputeResidualVectorRecommendedSYCLVectorSize(const int blksize,
								     const int team_size) {
      int total_team_size(0);
      if      (blksize <=  5) total_team_size =  32;
      else if (blksize <=  9) total_team_size =  32; // 64
      else if (blksize <= 12) total_team_size =  96;
      else if (blksize <= 16) total_team_size = 128;
      else if (blksize <= 20) total_team_size = 160;
      else                    total_team_size = 160;
      return total_team_size/team_size;
    }

    template<typename T>
    static inline int ComputeResidualVectorRecommendedVectorSize(const int blksize,
                                                                 const int team_size) {
      if ( is_cuda<T>::value )
        return ComputeResidualVectorRecommendedCudaVectorSize(blksize, team_size);
      if ( is_hip<T>::value )
        return ComputeResidualVectorRecommendedHIPVectorSize(blksize, team_size);
      if ( is_sycl<T>::value )
        return ComputeResidualVectorRecommendedSYCLVectorSize(blksize, team_size);
      return -1;
    }

    template<typename MatrixType>
    struct ComputeResidualVector {
    public:
      using impl_type = BlockHelperDetails::ImplType<MatrixType>;
      using node_device_type = typename impl_type::node_device_type;
      using execution_space = typename impl_type::execution_space;
      using memory_space = typename impl_type::memory_space;

      using local_ordinal_type = typename impl_type::local_ordinal_type;
      using size_type = typename impl_type::size_type;
      using impl_scalar_type = typename impl_type::impl_scalar_type;
      using magnitude_type = typename impl_type::magnitude_type;
      using btdm_scalar_type = typename impl_type::btdm_scalar_type;
      using btdm_magnitude_type = typename impl_type::btdm_magnitude_type;
      /// views
      using local_ordinal_type_1d_view = typename impl_type::local_ordinal_type_1d_view;
      using size_type_1d_view = typename impl_type::size_type_1d_view;
      using tpetra_block_access_view_type = typename impl_type::tpetra_block_access_view_type; // block crs (layout right)
      using impl_scalar_type_1d_view = typename impl_type::impl_scalar_type_1d_view;
      using impl_scalar_type_2d_view_tpetra = typename impl_type::impl_scalar_type_2d_view_tpetra; // block multivector (layout left)
      using vector_type_3d_view = typename impl_type::vector_type_3d_view;
      using btdm_scalar_type_4d_view = typename impl_type::btdm_scalar_type_4d_view;
      using i64_2d_view  = typename impl_type::i64_2d_view;
      static constexpr int vector_length = impl_type::vector_length;

      /// team policy member type (used in cuda)
      using member_type = typename Kokkos::TeamPolicy<execution_space>::member_type;

      // enum for max blocksize and vector length
      enum : int { max_blocksize = 32 };

    private:
      ConstUnmanaged<impl_scalar_type_2d_view_tpetra> b;
      ConstUnmanaged<impl_scalar_type_2d_view_tpetra> x; // x_owned
      ConstUnmanaged<impl_scalar_type_2d_view_tpetra> x_remote;
      Unmanaged<impl_scalar_type_2d_view_tpetra> y;
      Unmanaged<vector_type_3d_view> y_packed;
      Unmanaged<btdm_scalar_type_4d_view> y_packed_scalar;

      // AmD information
      const ConstUnmanaged<size_type_1d_view> rowptr, rowptr_remote;
      const ConstUnmanaged<local_ordinal_type_1d_view> colindsub, colindsub_remote;
      const ConstUnmanaged<impl_scalar_type_1d_view> tpetra_values;

      // block crs graph information
      // for cuda (kokkos crs graph uses a different size_type from size_t)
      const ConstUnmanaged<Kokkos::View<size_t*,node_device_type> > A_block_rowptr;
      const ConstUnmanaged<Kokkos::View<size_t*,node_device_type> > A_point_rowptr;
      const ConstUnmanaged<Kokkos::View<local_ordinal_type*,node_device_type> > A_colind;

      // blocksize
      const local_ordinal_type blocksize_requested;

      // part interface
      const ConstUnmanaged<local_ordinal_type_1d_view> part2packrowidx0;
      const ConstUnmanaged<local_ordinal_type_1d_view> part2rowidx0;
      const ConstUnmanaged<local_ordinal_type_1d_view> rowidx2part;
      const ConstUnmanaged<local_ordinal_type_1d_view> partptr;
      const ConstUnmanaged<local_ordinal_type_1d_view> lclrow;
      const ConstUnmanaged<local_ordinal_type_1d_view> dm2cm;

      // block offsets
      //const ConstUnmanaged<i64_2d_view> A_offsets;
      //const ConstUnmanaged<i64_2d_view> A_offsets_remote;
      //const ConstUnmanaged<i64_2d_view> x_offsets;
      //const ConstUnmanaged<i64_2d_view> x_offsets_remote;
      const i64_2d_view A_offsets;
      const i64_2d_view A_offsets_remote;
      const i64_2d_view x_offsets;
      const i64_2d_view x_offsets_remote;

      const bool is_dm2cm_active;
      const bool hasBlockCrsMatrix;

    public:
      template<typename LocalCrsGraphType>
      ComputeResidualVector(const AmD<MatrixType> &amd,
                            const LocalCrsGraphType &block_graph,
                            const LocalCrsGraphType &point_graph,
                            const local_ordinal_type &blocksize_requested_,
                            const PartInterface<MatrixType> &interf,
                            const local_ordinal_type_1d_view &dm2cm_,
                            bool hasBlockCrsMatrix_)
        : rowptr(amd.rowptr), rowptr_remote(amd.rowptr_remote),
          colindsub(amd.A_colindsub), colindsub_remote(amd.A_colindsub_remote),
          tpetra_values(amd.tpetra_values),
          A_block_rowptr(block_graph.row_map),
          A_point_rowptr(point_graph.row_map),
          A_colind(block_graph.entries),
          blocksize_requested(blocksize_requested_),
          part2packrowidx0(interf.part2packrowidx0),
          part2rowidx0(interf.part2rowidx0),
          rowidx2part(interf.rowidx2part),
          partptr(interf.partptr),
          lclrow(interf.lclrow),
          dm2cm(dm2cm_),
          A_offsets(amd.A_offsets),
          A_offsets_remote(amd.A_offsets_remote),
          x_offsets(amd.x_offsets),
          x_offsets_remote(amd.x_offsets_remote),
          is_dm2cm_active(dm2cm_.span() > 0),
          hasBlockCrsMatrix(hasBlockCrsMatrix_)
      {}

      inline
      void
      SerialDot(const local_ordinal_type &blocksize,
                const local_ordinal_type &lclRowID,
                const local_ordinal_type &lclColID,
                const local_ordinal_type &ii,
                const ConstUnmanaged<local_ordinal_type_1d_view>colindsub_,
                const impl_scalar_type * const KOKKOS_RESTRICT xx,
                /* */ impl_scalar_type * KOKKOS_RESTRICT yy) const {
        const size_type Aj_c = colindsub_(lclColID);
        auto point_row_offset = A_point_rowptr(lclRowID*blocksize + ii) + Aj_c*blocksize;
        impl_scalar_type val = 0;
#if defined(KOKKOS_ENABLE_PRAGMA_IVDEP)
#   pragma ivdep
#endif
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#   pragma unroll
#endif
        for (local_ordinal_type k1=0;k1<blocksize;++k1)
          val += tpetra_values(point_row_offset + k1) * xx[k1];
        yy[ii] -= val;
      }

      inline
      void
      SerialGemv(const local_ordinal_type &blocksize,
                 const impl_scalar_type * const KOKKOS_RESTRICT AA,
                 const impl_scalar_type * const KOKKOS_RESTRICT xx,
                 /* */ impl_scalar_type * KOKKOS_RESTRICT yy) const {
        using tlb = BlockHelperDetails::TpetraLittleBlock<Tpetra::Impl::BlockCrsMatrixLittleBlockArrayLayout>;
        for (local_ordinal_type k0=0;k0<blocksize;++k0) {
          impl_scalar_type val = 0;
#if defined(KOKKOS_ENABLE_PRAGMA_IVDEP)
#   pragma ivdep
#endif
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#   pragma unroll
#endif
          for (local_ordinal_type k1=0;k1<blocksize;++k1)
            val += AA[tlb::getFlatIndex(k0,k1,blocksize)]*xx[k1];
          yy[k0] -= val;
        }
      }

      template<typename bbViewType, typename yyViewType>
      KOKKOS_INLINE_FUNCTION
      void
      VectorCopy(const member_type &member,
                 const local_ordinal_type &blocksize,
                 const bbViewType &bb,
                 const yyViewType &yy) const {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, blocksize), [&](const local_ordinal_type &k0)  {
            yy(k0) = static_cast<typename yyViewType::const_value_type>(bb(k0));
          });
      }

      template<typename xxViewType, typename yyViewType>
      KOKKOS_INLINE_FUNCTION
      void
      VectorDot(const member_type &member,
                 const local_ordinal_type &blocksize,
                 const local_ordinal_type &lclRowID,
                 const local_ordinal_type &lclColID,
                 const local_ordinal_type &ii,
                 const ConstUnmanaged<local_ordinal_type_1d_view>colindsub_,
                 const xxViewType &xx,
                 const yyViewType &yy) const {
        const size_type Aj_c = colindsub_(lclColID);
        auto point_row_offset = A_point_rowptr(lclRowID*blocksize + ii) + Aj_c*blocksize;
        impl_scalar_type val = 0;
        Kokkos::parallel_reduce
          (Kokkos::ThreadVectorRange(member, blocksize),
           [&](const local_ordinal_type &k1, impl_scalar_type& update) {
              update += tpetra_values(point_row_offset + k1) *xx(k1);
          }, val);
        Kokkos::single(Kokkos::PerThread(member),
          [=]()
          {
            Kokkos::atomic_add(&yy(ii), typename yyViewType::const_value_type(-val));
          });
      }

      // BMK: This version coalesces accesses to AA for LayoutRight blocks.
      template<typename AAViewType, typename xxViewType, typename yyViewType>
      KOKKOS_INLINE_FUNCTION
      void
      VectorGemv(const member_type &member,
                 const local_ordinal_type &blocksize,
                 const AAViewType &AA,
                 const xxViewType &xx,
                 const yyViewType &yy) const {
         for (local_ordinal_type k0=0;k0<blocksize;++k0) {
           impl_scalar_type val = 0;
           Kokkos::parallel_reduce
             (Kokkos::ThreadVectorRange(member, blocksize),
              [&](const local_ordinal_type &k1, impl_scalar_type& update) {
                update += AA(k0,k1)*xx(k1);
             }, val);
           Kokkos::single(Kokkos::PerThread(member),
             [=]()
             {
               Kokkos::atomic_add(&yy(k0), -val);
             });
         }
      }

      struct SeqTag {};

      KOKKOS_INLINE_FUNCTION
      void
      operator() (const SeqTag &, const local_ordinal_type& i) const {
        const local_ordinal_type blocksize = blocksize_requested;
        const local_ordinal_type blocksize_square = blocksize*blocksize;

        // constants
        const Kokkos::pair<local_ordinal_type,local_ordinal_type> block_range(0, blocksize);
        const local_ordinal_type num_vectors = y.extent(1);
        const local_ordinal_type row = i*blocksize;
        for (local_ordinal_type col=0;col<num_vectors;++col) {
          // y := b
          impl_scalar_type *yy = &y(row, col);
          const impl_scalar_type * const bb = &b(row, col);
          memcpy(yy, bb, sizeof(impl_scalar_type)*blocksize);

          // y -= Rx
          const size_type A_k0 = A_block_rowptr[i];
          for (size_type k=rowptr[i];k<rowptr[i+1];++k) {
            const size_type j = A_k0 + colindsub[k];
            const impl_scalar_type * const xx = &x(A_colind[j]*blocksize, col);
            if (hasBlockCrsMatrix) {
              const impl_scalar_type * const AA = &tpetra_values(j*blocksize_square);
              SerialGemv(blocksize,AA,xx,yy);
            }
            else {
              for (local_ordinal_type k0=0;k0<blocksize;++k0)
                SerialDot(blocksize, i, k, k0, colindsub, xx, yy);
            }
          }
        }
      }

      KOKKOS_INLINE_FUNCTION
      void
      operator() (const SeqTag &, const member_type &member) const {

        // constants
        const local_ordinal_type blocksize = blocksize_requested;
        const local_ordinal_type blocksize_square = blocksize*blocksize;

        const local_ordinal_type lr = member.league_rank();
        const Kokkos::pair<local_ordinal_type,local_ordinal_type> block_range(0, blocksize);
        const local_ordinal_type num_vectors = y.extent(1);

        // subview pattern
        auto bb = Kokkos::subview(b, block_range, 0);
        auto xx = bb;
        auto yy = Kokkos::subview(y, block_range, 0);
        auto A_block_cst = ConstUnmanaged<tpetra_block_access_view_type>(NULL, blocksize, blocksize);

        const local_ordinal_type row = lr*blocksize;
        for (local_ordinal_type col=0;col<num_vectors;++col) {
          // y := b
          yy.assign_data(&y(row, col));
          bb.assign_data(&b(row, col));
          if (member.team_rank() == 0)
            VectorCopy(member, blocksize, bb, yy);
          member.team_barrier();

          // y -= Rx
          const size_type A_k0 = A_block_rowptr[lr];

          if(hasBlockCrsMatrix) {
            Kokkos::parallel_for
              (Kokkos::TeamThreadRange(member, rowptr[lr], rowptr[lr+1]),
              [&](const local_ordinal_type &k) {
                const size_type j = A_k0 + colindsub[k];
                xx.assign_data( &x(A_colind[j]*blocksize, col) );
                A_block_cst.assign_data( &tpetra_values(j*blocksize_square) );
                VectorGemv(member, blocksize, A_block_cst, xx, yy);
              });
          }
          else {
            Kokkos::parallel_for
              (Kokkos::TeamThreadRange(member, rowptr[lr], rowptr[lr+1]),
              [&](const local_ordinal_type &k) {

                const size_type j = A_k0 + colindsub[k];
                xx.assign_data( &x(A_colind[j]*blocksize, col) );

                for (local_ordinal_type k0=0;k0<blocksize;++k0)
                  VectorDot(member, blocksize, lr, k, k0, colindsub, xx, yy);
              });
          }
        }
      }

      // * B: block size for compile-time specialization, or 0 for general case (up to max_blocksize)
      // * async: true if using async importer. overlap is not used in this case.
      //          Whether a column is owned or nonowned is decided at runtime.
      // * overlap: true if processing the columns that are not locally owned,
      //            false if processing locally owned columns.
      // * haveBlockMatrix: true if A is a BlockCrsMatrix, false if it's CrsMatrix.
      template<int B, bool async, bool overlap, bool haveBlockMatrix>
      struct GeneralTag
      {
        static_assert(!(async && overlap),
            "ComputeResidualVector: async && overlap is not a valid configuration for GeneralTag");
      };

      // Define AsyncTag and OverlapTag in terms of GeneralTag:
      // P == 0 means only compute on owned columns
      // P == 1 means only compute on nonowned columns
      template<int P, int B, bool haveBlockMatrix>
      using OverlapTag = GeneralTag<B, false, P != 0, haveBlockMatrix>;

      template<int B, bool haveBlockMatrix>
      using AsyncTag = GeneralTag<B, true, false, haveBlockMatrix>;

      // CPU implementation for all cases
      template<int B, bool async, bool overlap, bool haveBlockMatrix>
      KOKKOS_INLINE_FUNCTION
      void
      operator() (const GeneralTag<B, async, overlap, haveBlockMatrix>&, const local_ordinal_type &rowidx) const {
        const local_ordinal_type blocksize = (B == 0 ? blocksize_requested : B);
        const local_ordinal_type blocksize_square = blocksize*blocksize;

        // constants
        const local_ordinal_type partidx = rowidx2part(rowidx);
        const local_ordinal_type pri = part2packrowidx0(partidx) + (rowidx - partptr(partidx));
        const local_ordinal_type v = partidx % vector_length;

        const local_ordinal_type num_vectors = y_packed.extent(2);
        const local_ordinal_type num_local_rows = lclrow.extent(0);

        // temporary buffer for y flat
        impl_scalar_type yy[B == 0 ? max_blocksize : B] = {};

        const local_ordinal_type lr = lclrow(rowidx);
        const local_ordinal_type row = lr*blocksize;

        auto colindsub_used = overlap ? colindsub_remote : colindsub;
        auto rowptr_used = overlap ? rowptr_remote : rowptr;

        for (local_ordinal_type col = 0; col < num_vectors; ++col) {
          if constexpr(overlap) {
            // y (temporary) := 0
            memset((void*) yy, 0, sizeof(impl_scalar_type)*blocksize);
          }
          else {
            // y := b
            memcpy(yy, &b(row, col), sizeof(impl_scalar_type)*blocksize);
          }

          // y -= Rx
          const size_type A_k0 = A_block_rowptr[lr];
          for (size_type k = rowptr_used[lr]; k < rowptr_used[lr+1]; ++k) {
            const size_type j = A_k0 + colindsub_used[k];
            const local_ordinal_type A_colind_at_j = A_colind[j];
            if constexpr (haveBlockMatrix) {
              /*
              if(async) {
                // Check that precomputed offset matches what we got here
                size_type precomputed = A_offsets(rowidx, k - rowptr_used[lr]);
                size_type actual = j * blocksize_square;
                if(actual != precomputed)
                  printf("Error async: A precomputed offset incorrect (%d vs %d)\n", (int) precomputed, (int) actual);
              }
              else {
                if(overlap) {
                  size_type precomputed = A_offsets_remote(rowidx, k - rowptr_used[lr]);
                  size_type actual = j * blocksize_square;
                  if(actual != precomputed)
                    printf("Error overlap remote: A precomputed offset incorrect (%d vs %d)\n", (int) precomputed, (int) actual);
                }
                else {
                  size_type precomputed = A_offsets(rowidx, k - rowptr_used[lr]);
                  size_type actual = j * blocksize_square;
                  if(actual != precomputed)
                    printf("Error overlap owned: A precomputed offset incorrect (%d vs %d)\n", (int) precomputed, (int) actual);
                }
              }
              */

              const impl_scalar_type * const AA = &tpetra_values(j*blocksize_square);
              if ((!async && !overlap) || (async && A_colind_at_j < num_local_rows)) {
                const auto loc = is_dm2cm_active ? dm2cm[A_colind_at_j] : A_colind_at_j;
                const impl_scalar_type * const xx = &x(loc*blocksize, col);
                SerialGemv(blocksize, AA,xx,yy);

                size_type actual = loc * blocksize;
                /*
                if(async) {
                  size_type precomputed = x_offsets(rowidx, k - rowptr_used[lr]);
                  if(actual != precomputed)
                    printf("Error async owned: x precomputed offset incorrect (%d vs %d)\n", (int) precomputed, (int) actual);
                }
                else {
                  size_type precomputed = x_offsets(rowidx, k - rowptr_used[lr]);
                  if(actual != precomputed)
                    printf("Error overlap owned: x precomputed offset incorrect (%d vs %d)\n", (int) precomputed, (int) actual);
                }
                */
              } else {
                const auto loc = A_colind_at_j - num_local_rows;
                const impl_scalar_type * const xx_remote = &x_remote(loc*blocksize, col);
                SerialGemv(blocksize, AA,xx_remote,yy);

                size_type actual = loc * blocksize;
                /*
                if(async) {
                  size_type precomputed = x_offsets(rowidx, k - rowptr_used[lr]) - (blocksize * num_local_rows);
                  if(actual != precomputed)
                    printf("Error async remote: x precomputed offset incorrect (%d vs %d)\n", (int) precomputed, (int) actual);
                }
                else {
                  size_type precomputed = x_offsets_remote(rowidx, k - rowptr_used[lr]);
                  if(actual != precomputed)
                    printf("Error overlap remote: x precomputed offset incorrect (%d vs %d)\n", (int) precomputed, (int) actual);
                }
                */
              }
            }
            else {
              if ((!async && !overlap) || (async && A_colind_at_j < num_local_rows)) {
                const auto loc = is_dm2cm_active ? dm2cm[A_colind_at_j] : A_colind_at_j;
                const impl_scalar_type * const xx = &x(loc*blocksize, col);
                for (local_ordinal_type k0 = 0; k0 < blocksize; ++k0)
                  SerialDot(blocksize, lr, k, k0, colindsub_used, xx, yy);
              } else {
                const auto loc = A_colind_at_j - num_local_rows;
                const impl_scalar_type * const xx_remote = &x_remote(loc*blocksize, col);
                for (local_ordinal_type k0 = 0; k0 < blocksize; ++k0)
                  SerialDot(blocksize, lr, k, k0, colindsub_used, xx_remote, yy);
              }
            }
          }
          // move yy to y_packed
          if constexpr(overlap) {
            for (local_ordinal_type k=0;k<blocksize;++k)
              y_packed(pri, k, col)[v] += yy[k];
          }
          else {
            for (local_ordinal_type k=0;k<blocksize;++k)
              y_packed(pri, k, col)[v] = yy[k];
          }
        }
      }

      // GPU implementation for hasBlockCrsMatrix == true
      template<int B, bool async, bool overlap>
      KOKKOS_INLINE_FUNCTION
      void
      operator() (const GeneralTag<B, async, overlap, true>&, const member_type &member) const {
        const local_ordinal_type blocksize = (B == 0 ? blocksize_requested : B);
        const local_ordinal_type blocksize_square = blocksize*blocksize;

        // constants
        const local_ordinal_type rowidx = member.league_rank();
        const local_ordinal_type partidx = rowidx2part(rowidx);
        const local_ordinal_type pri = part2packrowidx0(partidx) + (rowidx - partptr(partidx));
        const local_ordinal_type v = partidx % vector_length;

        const Kokkos::pair<local_ordinal_type,local_ordinal_type> block_range(0, blocksize);
        const local_ordinal_type num_vectors = y_packed_scalar.extent(2);
        const local_ordinal_type num_local_rows = lclrow.extent(0);

        // subview pattern
        using subview_1D_right_t = decltype(Kokkos::subview(b, block_range, 0));
        using subview_1D_stride_t = decltype(Kokkos::subview(y_packed_scalar, 0, block_range, 0, 0));
        subview_1D_right_t bb(nullptr, blocksize);
        subview_1D_right_t xx(nullptr, blocksize);
        subview_1D_stride_t yy(nullptr, Kokkos::LayoutStride(blocksize, y_packed_scalar.stride_1()));
        auto A_block_cst = ConstUnmanaged<tpetra_block_access_view_type>(NULL, blocksize, blocksize);

        // Get shared allocation for a local copy of x, Ax, and A
        impl_scalar_type * local_Ax = reinterpret_cast<impl_scalar_type *>(member.team_scratch(0).get_shmem(blocksize*sizeof(impl_scalar_type)));
        impl_scalar_type * local_x = reinterpret_cast<impl_scalar_type *>(member.thread_scratch(0).get_shmem(blocksize*sizeof(impl_scalar_type)));

        const local_ordinal_type lr = lclrow(rowidx);
        const local_ordinal_type row = lr*blocksize;
        for (local_ordinal_type col = 0; col < num_vectors; ++col) {
          if(col)
            member.team_barrier();
          // y -= Rx
          // Initialize accumulation array
          Kokkos::parallel_for(Kokkos::TeamVectorRange(member, blocksize),[&](const local_ordinal_type & i){
            local_Ax[i] = 0;
          });
          member.team_barrier();

          int numEntries;
          if constexpr(!overlap) {
            numEntries = A_offsets.extent(1);
          }
          else {
            numEntries = A_offsets_remote.extent(1);
          }

          Kokkos::parallel_for
            (Kokkos::TeamThreadRange(member, 0, numEntries),
            [&](const int k) {
              int64_t A_offset = overlap ? A_offsets_remote(rowidx, k) : A_offsets(rowidx, k);
              if(A_offset != Kokkos::ArithTraits<int64_t>::min()) {
                A_block_cst.assign_data(tpetra_values.data() + A_offset);
                // Pull x into local memory
                if constexpr(async) {
                  int64_t x_offset = x_offsets(rowidx, k);
                  size_type remote_cutoff = blocksize * num_local_rows;
                  if(x_offset >= remote_cutoff)
                    xx.assign_data(&x_remote(x_offset - remote_cutoff, col));
                  else
                    xx.assign_data(&x(x_offset, col));
                }
                else {
                  if constexpr(!overlap) {
                    xx.assign_data(&x(x_offsets(rowidx, k), col));
                  }
                  else {
                    xx.assign_data(&x_remote(x_offsets(rowidx, k), col));
                  }
                }

                Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, blocksize),[&](const local_ordinal_type & i){
                  local_x[i] = xx(i);
                });
              
                // MatVec op Ax += A*x
                Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(member, blocksize),
                  [&](const local_ordinal_type &k0) {
                    impl_scalar_type val = 0;
                    for(int k1=0; k1<blocksize; k1++)
                      val += A_block_cst(k0,k1)*local_x[k1];
                    Kokkos::atomic_add(local_Ax+k0, val);
                });
              }
            });
          member.team_barrier();
          // Update y = b - local_Ax
          yy.assign_data(&y_packed_scalar(pri, 0, col, v));
          bb.assign_data(&b(row, col));
          Kokkos::parallel_for(Kokkos::TeamVectorRange(member, blocksize),[&](const local_ordinal_type & i){
            if (!overlap)
              yy(i) = bb(i) - local_Ax[i];
            else
              yy(i) -= local_Ax[i];
          });
        }
      }

      // GPU implementation for hasBlockCrsMatrix == false
      template<int B, bool async, bool overlap>
      KOKKOS_INLINE_FUNCTION
      void
      operator() (const GeneralTag<B, async, overlap, false>&, const member_type &member) const {
        const local_ordinal_type blocksize = (B == 0 ? blocksize_requested : B);
        const local_ordinal_type blocksize_square = blocksize*blocksize;

        // constants
        const local_ordinal_type rowidx = member.league_rank();
        const local_ordinal_type partidx = rowidx2part(rowidx);
        const local_ordinal_type pri = part2packrowidx0(partidx) + (rowidx - partptr(partidx));
        const local_ordinal_type v = partidx % vector_length;

        const Kokkos::pair<local_ordinal_type,local_ordinal_type> block_range(0, blocksize);
        const local_ordinal_type num_vectors = y_packed_scalar.extent(2);
        const local_ordinal_type num_local_rows = lclrow.extent(0);

        // subview pattern
        using subview_1D_right_t = decltype(Kokkos::subview(b, block_range, 0));
        using subview_1D_stride_t = decltype(Kokkos::subview(y_packed_scalar, 0, block_range, 0, 0));
        subview_1D_right_t bb(nullptr, blocksize);
        subview_1D_right_t xx(nullptr, blocksize);
        subview_1D_right_t xx_remote(nullptr, blocksize);
        subview_1D_stride_t yy(nullptr, Kokkos::LayoutStride(blocksize, y_packed_scalar.stride_1()));
        auto A_block_cst = ConstUnmanaged<tpetra_block_access_view_type>(NULL, blocksize, blocksize);
        auto colindsub_used = overlap ? colindsub_remote : colindsub;
        auto rowptr_used = overlap ? rowptr_remote : rowptr;

        const local_ordinal_type lr = lclrow(rowidx);
        const local_ordinal_type row = lr*blocksize;
        for (local_ordinal_type col = 0; col < num_vectors; ++col) {
          yy.assign_data(&y_packed_scalar(pri, 0, col, v));
          if (!overlap) {
            // y := b
            bb.assign_data(&b(row, col));
            if (member.team_rank() == 0)
              VectorCopy(member, blocksize, bb, yy);
            member.team_barrier();
          }

          // y -= Rx
          const size_type A_k0 = A_block_rowptr[lr];
          Kokkos::parallel_for
            (Kokkos::TeamThreadRange(member, rowptr_used[lr], rowptr_used[lr+1]),
            [&](const local_ordinal_type &k) {
              const size_type j = A_k0 + colindsub_used[k];
              const local_ordinal_type A_colind_at_j = A_colind[j];
              if ((async && A_colind_at_j < num_local_rows) || (!async && !overlap)) {
                const auto loc = is_dm2cm_active ? dm2cm[A_colind_at_j] : A_colind_at_j;
                xx.assign_data( &x(loc*blocksize, col) );
                for (local_ordinal_type k0 = 0; k0 < blocksize; ++k0)
                  VectorDot(member, blocksize, lr, k, k0, colindsub_used, xx, yy);
              } else {
                const auto loc = A_colind_at_j - num_local_rows;
                xx_remote.assign_data( &x_remote(loc*blocksize, col) );
                for (local_ordinal_type k0 = 0; k0 < blocksize; ++k0)
                  VectorDot(member, blocksize, lr, k, k0, colindsub_used, xx_remote, yy);
              }
            });
        }
      }

      // y = b - Rx; seq method
      template<typename MultiVectorLocalViewTypeY,
               typename MultiVectorLocalViewTypeB,
               typename MultiVectorLocalViewTypeX>
      void run(const MultiVectorLocalViewTypeY &y_,
               const MultiVectorLocalViewTypeB &b_,
               const MultiVectorLocalViewTypeX &x_) {
        IFPACK2_BLOCKHELPER_PROFILER_REGION_BEGIN;
        IFPACK2_BLOCKHELPER_TIMER_WITH_FENCE("BlockTriDi::ComputeResidual::<SeqTag>", ComputeResidual0, execution_space);

        y = y_; b = b_; x = x_;
        if constexpr (is_device<execution_space>::value) {
          const local_ordinal_type blocksize = blocksize_requested;
          const local_ordinal_type team_size = 8;
          const local_ordinal_type vector_size = ComputeResidualVectorRecommendedVectorSize<execution_space>(blocksize, team_size);
          const Kokkos::TeamPolicy<execution_space,SeqTag> policy(rowptr.extent(0) - 1, team_size, vector_size);
          Kokkos::parallel_for
            ("ComputeResidual::TeamPolicy::run<SeqTag>", policy, *this);
        } else {
          const Kokkos::RangePolicy<execution_space,SeqTag> policy(0, rowptr.extent(0) - 1);
          Kokkos::parallel_for
            ("ComputeResidual::RangePolicy::run<SeqTag>", policy, *this);
        }
        IFPACK2_BLOCKHELPER_PROFILER_REGION_END;
        IFPACK2_BLOCKHELPER_TIMER_FENCE(execution_space)
      }

      // y = b - R (x , x_remote)
      template<typename MultiVectorLocalViewTypeB,
               typename MultiVectorLocalViewTypeX,
               typename MultiVectorLocalViewTypeX_Remote>
      void run(const vector_type_3d_view &y_packed_,
               const MultiVectorLocalViewTypeB &b_,
               const MultiVectorLocalViewTypeX &x_,
               const MultiVectorLocalViewTypeX_Remote &x_remote_) {
        IFPACK2_BLOCKHELPER_PROFILER_REGION_BEGIN;
        IFPACK2_BLOCKHELPER_TIMER_WITH_FENCE("BlockTriDi::ComputeResidual::<AsyncTag>", ComputeResidual0, execution_space);

        b = b_; x = x_; x_remote = x_remote_;
        if constexpr (is_device<execution_space>::value) {
          y_packed_scalar = btdm_scalar_type_4d_view((btdm_scalar_type*)y_packed_.data(),
                                                     y_packed_.extent(0),
                                                     y_packed_.extent(1),
                                                     y_packed_.extent(2),
                                                     vector_length);
        } else {
          y_packed = y_packed_;
        }

        if constexpr(is_device<execution_space>::value) {
          const local_ordinal_type blocksize = blocksize_requested;
          // local_ordinal_type vl_power_of_two = 1;
          // for (;vl_power_of_two<=blocksize_requested;vl_power_of_two*=2);
          // vl_power_of_two *= (vl_power_of_two < blocksize_requested ? 2 : 1);
          // const local_ordinal_type vl = vl_power_of_two > vector_length ? vector_length : vl_power_of_two;
#define BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(B) {                \
            if(this->hasBlockCrsMatrix) { \
              const local_ordinal_type team_size = 8; \
              const local_ordinal_type vector_size = 8; \
              const size_t shmem_team_size = blocksize*sizeof(btdm_scalar_type); \
              const size_t shmem_thread_size = blocksize*sizeof(btdm_scalar_type); \
              Kokkos::TeamPolicy<execution_space,AsyncTag<B, true> >      \
                policy(rowidx2part.extent(0), team_size, vector_size);    \
              policy.set_scratch_size(0,Kokkos::PerTeam(shmem_team_size),Kokkos::PerThread(shmem_thread_size)); \
              Kokkos::parallel_for                                        \
                ("ComputeResidual::TeamPolicy::run<AsyncTag>",            \
                 policy, *this); \
            } \
            else { \
              const local_ordinal_type team_size = 8; \
              const local_ordinal_type vector_size = ComputeResidualVectorRecommendedVectorSize<execution_space>(blocksize, team_size); \
              const Kokkos::TeamPolicy<execution_space,AsyncTag<B, false> >      \
                policy(rowidx2part.extent(0), team_size, vector_size);    \
              Kokkos::parallel_for                                        \
                ("ComputeResidual::TeamPolicy::run<AsyncTag>",            \
                 policy, *this); \
            } \
          } break
          switch (blocksize_requested) {
          case   3: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 3);
          case   5: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 5);
          case   7: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 7);
          case   9: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 9);
          case  10: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(10);
          case  11: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(11);
          case  16: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(16);
          case  17: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(17);
          case  18: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(18);
          default : BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 0);
          }
#undef BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL
	} else {
#define BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(B) {                \
            if(this->hasBlockCrsMatrix) { \
              const Kokkos::RangePolicy<execution_space,AsyncTag<B, true> > policy(0, rowidx2part.extent(0)); \
              Kokkos::parallel_for                                        \
                ("ComputeResidual::RangePolicy::run<AsyncTag>",           \
                 policy, *this); \
            } \
            else { \
              const Kokkos::RangePolicy<execution_space,AsyncTag<B, false> > policy(0, rowidx2part.extent(0)); \
              Kokkos::parallel_for                                        \
                ("ComputeResidual::RangePolicy::run<AsyncTag>",           \
                 policy, *this); \
            } \
          } break

          switch (blocksize_requested) {
          case   3: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 3);
          case   5: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 5);
          case   7: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 7);
          case   9: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 9);
          case  10: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(10);
          case  11: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(11);
          case  16: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(16);
          case  17: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(17);
          case  18: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(18);
          default : BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 0);
          }
#undef BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL
        }
        IFPACK2_BLOCKHELPER_PROFILER_REGION_END;
        IFPACK2_BLOCKHELPER_TIMER_FENCE(execution_space)
      }

      // y = b - R (y , y_remote)
      template<typename MultiVectorLocalViewTypeB,
               typename MultiVectorLocalViewTypeX,
               typename MultiVectorLocalViewTypeX_Remote>
      void run(const vector_type_3d_view &y_packed_,
               const MultiVectorLocalViewTypeB &b_,
               const MultiVectorLocalViewTypeX &x_,
               const MultiVectorLocalViewTypeX_Remote &x_remote_,
               const bool compute_owned) {
        IFPACK2_BLOCKHELPER_PROFILER_REGION_BEGIN;
        IFPACK2_BLOCKHELPER_TIMER_WITH_FENCE("BlockTriDi::ComputeResidual::<OverlapTag>", ComputeResidual0, execution_space);

        b = b_; x = x_; x_remote = x_remote_;
        if constexpr (is_device<execution_space>::value) {
          y_packed_scalar = btdm_scalar_type_4d_view((btdm_scalar_type*)y_packed_.data(),
                                                     y_packed_.extent(0),
                                                     y_packed_.extent(1),
                                                     y_packed_.extent(2),
                                                     vector_length);
        } else {
          y_packed = y_packed_;
        }

        if constexpr (is_device<execution_space>::value) {
          const local_ordinal_type blocksize = blocksize_requested;
          // local_ordinal_type vl_power_of_two = 1;
          // for (;vl_power_of_two<=blocksize_requested;vl_power_of_two*=2);
          // vl_power_of_two *= (vl_power_of_two < blocksize_requested ? 2 : 1);
          // const local_ordinal_type vl = vl_power_of_two > vector_length ? vector_length : vl_power_of_two;
#define BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(B)  \
          if(this->hasBlockCrsMatrix) { \
            const local_ordinal_type team_size = 8; \
            const local_ordinal_type vector_size = 8; \
            const size_t shmem_team_size = blocksize*sizeof(btdm_scalar_type); \
            const size_t shmem_thread_size = blocksize*sizeof(btdm_scalar_type); \
            if (compute_owned) {                                          \
              Kokkos::TeamPolicy<execution_space,OverlapTag<0, B, true> > \
                policy(rowidx2part.extent(0), team_size, vector_size);    \
              policy.set_scratch_size(0,Kokkos::PerTeam(shmem_team_size),Kokkos::PerThread(shmem_thread_size)); \
              Kokkos::parallel_for                                        \
                ("ComputeResidual::TeamPolicy::run<OverlapTag<0> >", policy, *this); \
            } else {                                                      \
              Kokkos::TeamPolicy<execution_space,OverlapTag<1, B, true> > \
                policy(rowidx2part.extent(0), team_size, vector_size);    \
              policy.set_scratch_size(0,Kokkos::PerTeam(shmem_team_size),Kokkos::PerThread(shmem_thread_size)); \
              Kokkos::parallel_for                                        \
                ("ComputeResidual::TeamPolicy::run<OverlapTag<1> >", policy, *this); \
            } \
          } \
          else { \
            const local_ordinal_type team_size = 8; \
            const local_ordinal_type vector_size = ComputeResidualVectorRecommendedVectorSize<execution_space>(blocksize, team_size); \
            if (compute_owned) {                                          \
              const Kokkos::TeamPolicy<execution_space,OverlapTag<0, B, false> > \
                policy(rowidx2part.extent(0), team_size, vector_size);    \
              Kokkos::parallel_for                                        \
                ("ComputeResidual::TeamPolicy::run<OverlapTag<0> >", policy, *this); \
            } else {                                                      \
              const Kokkos::TeamPolicy<execution_space,OverlapTag<1, B, false> > \
                policy(rowidx2part.extent(0), team_size, vector_size);    \
              Kokkos::parallel_for                                        \
                ("ComputeResidual::TeamPolicy::run<OverlapTag<1> >", policy, *this); \
            } \
          } \
          break
          switch (blocksize_requested) {
          case   3: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 3);
          case   5: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 5);
          case   7: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 7);
          case   9: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 9);
          case  10: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(10);
          case  11: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(11);
          case  16: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(16);
          case  17: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(17);
          case  18: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(18);
          default : BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 0);
          }
#undef BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL
        } else {
#define BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(B)  \
          if(this->hasBlockCrsMatrix) { \
            if (compute_owned) {                                          \
              const Kokkos::RangePolicy<execution_space,OverlapTag<0, B, true>> \
                policy(0, rowidx2part.extent(0));                         \
              Kokkos::parallel_for                                        \
                ("ComputeResidual::RangePolicy::run<OverlapTag<0> >", policy, *this); \
            } else {                                                      \
              const Kokkos::RangePolicy<execution_space,OverlapTag<1, B, true>> \
                policy(0, rowidx2part.extent(0));                         \
              Kokkos::parallel_for                                        \
                ("ComputeResidual::RangePolicy::run<OverlapTag<1> >", policy, *this); \
            } \
          } \
          else { \
            if (compute_owned) {                                          \
              const Kokkos::RangePolicy<execution_space,OverlapTag<0, B, false>> \
                policy(0, rowidx2part.extent(0));                         \
              Kokkos::parallel_for                                        \
                ("ComputeResidual::RangePolicy::run<OverlapTag<0> >", policy, *this); \
            } else {                                                      \
              const Kokkos::RangePolicy<execution_space,OverlapTag<1, B, false>> \
                policy(0, rowidx2part.extent(0));                         \
              Kokkos::parallel_for                                        \
                ("ComputeResidual::RangePolicy::run<OverlapTag<1> >", policy, *this); \
            } \
          } \
          break

          switch (blocksize_requested) {
          case   3: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 3);
          case   5: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 5);
          case   7: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 7);
          case   9: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 9);
          case  10: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(10);
          case  11: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(11);
          case  16: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(16);
          case  17: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(17);
          case  18: BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL(18);
          default : BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL( 0);
          }
#undef BLOCKTRIDICONTAINER_DETAILS_COMPUTERESIDUAL
        }
        IFPACK2_BLOCKHELPER_PROFILER_REGION_END;
        IFPACK2_BLOCKHELPER_TIMER_FENCE(execution_space)
      }
    };


  } // namespace BlockHelperDetails

} // namespace Ifpack2

#endif
