// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/core/cuda.hpp"

#ifndef HAVE_CUDA

namespace opencv_test { namespace {

class GpuMatReleaseAllocator : public cv::cuda::GpuMat::Allocator
{
public:
    GpuMatReleaseAllocator() : freeCount(0) {}

    bool allocate(cv::cuda::GpuMat*, int, int, size_t) CV_OVERRIDE
    {
        return false;
    }

    void free(cv::cuda::GpuMat*) CV_OVERRIDE
    {
        ++freeCount;
    }

    int freeCount;
};

TEST(Core_GpuMat, release_uses_header_allocator_without_cuda)
{
    GpuMatReleaseAllocator allocator;
    uchar storage[6] = {};
    int refcount = 1;
    cv::cuda::GpuMat first(2, 3, CV_8UC1, storage);
    first.allocator = &allocator;
    first.refcount = &refcount;
    cv::cuda::GpuMat second = first;

    ASSERT_EQ(2, refcount);
    first.release();
    EXPECT_EQ(1, refcount);
    EXPECT_EQ(0, allocator.freeCount);
    EXPECT_TRUE(first.empty());

    second.release();
    EXPECT_EQ(0, refcount);
    EXPECT_EQ(1, allocator.freeCount);
    EXPECT_TRUE(second.empty());

    first.release();
    second.release();
    EXPECT_EQ(1, allocator.freeCount);
}

TEST(Core_GpuMat, release_external_header_without_cuda)
{
    uchar storage[6] = {};
    cv::cuda::GpuMat mat(2, 3, CV_8UC1, storage);
    ASSERT_EQ(nullptr, mat.refcount);

    mat.release();
    EXPECT_TRUE(mat.empty());
    EXPECT_EQ(nullptr, mat.data);
    EXPECT_EQ(nullptr, mat.refcount);
    mat.release();
}

}} // namespace

#endif // HAVE_CUDA
