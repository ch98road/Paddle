// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/inference/api/infer_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/core/dense_tensor.h"
#ifdef PADDLE_WITH_XPU
#include "xpu/runtime.h"
#endif
#include "glog/logging.h"

namespace paddle {

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
InferGPUContext::InferGPUContext(const phi::Place& place)
    : phi::GPUContext(place, false) {}
#endif

#ifdef PADDLE_WITH_XPU
InferXPUContext::InferXPUContext(const phi::Place& place, int context_gm_size)
    : phi::XPUContext(place) {
  if (context_gm_size >= 0) {
    x_context()->set_option("XPUAPI_DEFAULT_SIZE",
                            std::to_string(context_gm_size).c_str());
  } else {
    x_context()->set_option("XPUAPI_DEFAULT_SIZE", "");
  }
}

/**
 * @brief 分配内存空间
 *
 * 根据给定的张量、数据类型、请求的大小、是否固定以及是否进行假分配，在InferXPU上下文中分配内存空间。
 *
 * @param tensor 张量指针
 * @param dtype 数据类型
 * @param requested_size 请求的大小
 * @param pinned 是否固定内存
 * @param fake_alloc 是否进行假分配
 *
 * @return 分配的内存空间指针
 */
// InferXPUContext类的成员函数，用于分配内存
// 参数包括：tensor（张量基础对象），dtype（数据类型），requested_size（请求的内存大小），
// pinned（是否锁定内存，防止其被换出到磁盘），fake_alloc（是否进行假分配，即不实际分配内存，仅用于测试或模拟）
void* InferXPUContext::Alloc(phi::TensorBase* tensor,
                             phi::DataType dtype,
                             size_t requested_size,
                             bool pinned,
                             bool fake_alloc) const {
  VLOG(1) << "ch -- InferXPUContext::Alloc, l3_autotune_size_ = "
          << l3_autotune_size_;

  // 计算需要分配的内存大小，即张量的元素数量乘以每个元素的大小
  size_t size = tensor->numel() * phi::SizeOf(tensor->dtype());
  // 如果l3_autotune_size_大于0且holder_map_为空，则执行以下操作
  // holder_map_在InferXPUContext::L3CacheAutotune()中被设置，这意味着运行的第一次一定是空
  if (l3_autotune_size_ > 0 && holder_map_.empty()) {
    VLOG(1) << "ch -- InferXPUContext::Alloc before alloc";
    // 调用DeviceContext的Alloc函数进行实际的内存分配，返回分配的内存指针
    // 这里的alloc好像都是在GM上的
    void* data_ptr =
        DeviceContext::Alloc(tensor, dtype, requested_size, pinned, fake_alloc);
    VLOG(1) << "ch -- InferXPUContext::Alloc after alloc, data_ptr: "
            << data_ptr << ", size: " << size << ", numel: " << tensor->numel();

    // 初始化L3缓存块指针为nullptr
    phi::XPUL3CacheBlock* l3_block = nullptr;
    // 获取张量的持有者（holder）
    phi::Allocation* holder =
        reinterpret_cast<phi::DenseTensor*>(tensor)->Holder().get();
    VLOG(1) << "ch -- InferXPUContext::Alloc holder: " << holder->ptr()
            << ", size: " << holder->size();
    if (holder_l3_blocks_.count(holder) == 0) {
      VLOG(1)
          << "ch -- InferXPUContext::Alloc holder_l3_blocks_ haven't holder";

      l3_block = new phi::XPUL3CacheBlock();
      holder_l3_blocks_[holder] = l3_block;
      l3_blocks_.push_back(l3_block);
    } else {
      VLOG(1) << "ch -- InferXPUContext::Alloc holder_l3_blocks_ have holder";
      l3_block = holder_l3_blocks_[holder];
    }
    l3_block->Record(size);
    // VLOG(1) << "ch -- InferXPUContext::Alloc l3_block->size() = "
    //         << l3_block->size();
    return data_ptr;
  } else if (l3_autotune_size_ > 0 && !holder_map_.empty()) {
    phi::Allocation* holder =
        reinterpret_cast<phi::DenseTensor*>(tensor)->Holder().get();
    auto holder_iter = holder_map_.find(holder);
    VLOG(1) << "ch -- holder_map_.find(holder)";
    if (holder_iter != holder_map_.end()) {
      auto& holder_pair = holder_iter->second;
      auto* swap_holder = holder_pair.first;
      bool& swap_holder_is_l3 = holder_pair.second;

      // VLOG(1) << "ch -- holder_pair.first = " << holder_pair.first->ptr()
      //         << ", size " << holder_pair.first->size()
      //         << " swap_holder_is_l3 = " << swap_holder_is_l3
      //         << " swap_holder = " << swap_holder->ptr() << ", size "
      //         << swap_holder->size();

      // 如果size > 当前L3中的size，则使用GM中的地址，反之则使用L3的地址
      if (swap_holder_is_l3 && swap_holder->size() >= size) {
        swap(*holder, *swap_holder);
        swap_holder_is_l3 = false;
      } else if (!swap_holder_is_l3 && holder->size() < size) {
        swap(*holder, *swap_holder);
        swap_holder_is_l3 = true;
      }
    }
    void* data_ptr =
        DeviceContext::Alloc(tensor, dtype, requested_size, pinned, fake_alloc);
    return DeviceContext::Alloc(
        tensor, dtype, requested_size, pinned, fake_alloc);
  } else {
    return DeviceContext::Alloc(
        tensor, dtype, requested_size, pinned, fake_alloc);
  }
}

void InferXPUContext::SetXContext(xpu::Context* x_context) {
  VLOG(1) << "ch -- int SetXContext";
  auto* old_x_context = this->x_context();
  if (old_x_context != x_context) {
    if (l3_owned_ && l3_size_ > 0 &&
        (x_context->_l3_mgr.get_size() != l3_size_ ||
         x_context->_l3_mgr.get_ptr() != l3_ptr_)) {
      xpu_free(l3_ptr_);
    }
    old_x_context->_l3_mgr.set(nullptr, 0);
    l3_size_ = x_context->_l3_mgr.get_size();
    l3_ptr_ = x_context->_l3_mgr.get_ptr();
    l3_owned_ = false;
    phi::XPUContext::SetXContext(x_context);
  }
}

void InferXPUContext::SetL3Info(size_t l3_size,
                                void* l3_ptr,
                                size_t l3_autotune_size,
                                const phi::Place& place) {
  phi::backends::xpu::XPUDeviceGuard guard(place.GetDeviceId());
  if (l3_ptr == nullptr) {
    if (l3_size_ != l3_size) {
      if (l3_owned_) {
        xpu_free(l3_ptr_);
      }
      if (l3_size > 0) {
        xpu_malloc(&l3_ptr_, l3_size, XPU_MEM_L3);
        if (l3_ptr_ != nullptr) {
          VLOG(3) << "remalloc l3(" << l3_size << ") success.";
          VLOG(1) << "ch -- l3_ptr_:" << l3_ptr_;
          l3_size_ = l3_size;
          l3_owned_ = true;
          l3_autotune_size_ = l3_autotune_size;
        } else {
          VLOG(3) << "malloc l3(" << l3_size << ") failed. No l3 will be used.";
          l3_size_ = 0;
          l3_owned_ = false;
          l3_autotune_size_ = 0;
        }
      }
    }
  } else {
    if (l3_owned_) {
      xpu_free(l3_ptr_);
    }
    l3_ptr_ = l3_ptr;
    l3_size_ = l3_size;
    l3_autotune_size_ = l3_autotune_size;
  }
  if (l3_autotune_size_ == 0) {
    x_context()->_l3_mgr.set(l3_ptr_, l3_size_);
  }
}

void InferXPUContext::ClearL3Block(void* out_tensor_ptr) {
  for (auto& holder_l3_block : holder_l3_blocks_) {
    if (holder_l3_block.first->ptr() == out_tensor_ptr) {
      holder_l3_block.second->Clear();
    }
  }
}

void InferXPUContext::SetConvAutotuneInfo(std::string conv_autotune_file,
                                          int conv_autotune_level,
                                          bool conv_autotune_file_writeback,
                                          const phi::Place& place) {
  phi::backends::xpu::XPUDeviceGuard guard(place.GetDeviceId());

  VLOG(5) << "XPU conv autotune level:" << conv_autotune_level;
  VLOG(5) << "XPU conv autotune file:" << conv_autotune_file;
  VLOG(5) << "XPU conv autotune file writeback:"
          << conv_autotune_file_writeback;

  if (!conv_autotune_file.empty()) {
    int ret;
    ret = x_context()->set_option("XPU_CONV_AUTOTUNE_FILE",
                                  conv_autotune_file.c_str());
    PADDLE_ENFORCE_EQ(
        ret,
        0,
        platform::errors::Unavailable(
            "Failed to set XPU conv autotune file %s.", conv_autotune_file));
  }
  if (conv_autotune_level > 0) {
    int ret;
    ret = x_context()->set_option(
        "XPU_CONV_AUTOTUNE", (std::to_string(conv_autotune_level)).c_str());
    PADDLE_ENFORCE_EQ(
        ret,
        0,
        platform::errors::Unavailable("Failed to set XPU conv autotune  %d.",
                                      conv_autotune_level));
  }
  if (conv_autotune_file_writeback) {
    int ret;
    ret = x_context()->set_option(
        "XPU_AUTOTUNE_WRITEBACK",
        (std::to_string(conv_autotune_file_writeback)).c_str());
    PADDLE_ENFORCE_EQ(ret,
                      0,
                      platform::errors::Unavailable(
                          "Failed to set XPU conv autotune writeback %d.",
                          conv_autotune_file_writeback));
  }
}

void InferXPUContext::SetFcAutotuneInfo(std::string fc_autotune_file,
                                        int fc_autotune_level,
                                        bool fc_autotune_file_writeback,
                                        const phi::Place& place) {
  phi::backends::xpu::XPUDeviceGuard guard(place.GetDeviceId());

  VLOG(5) << "XPU fc autotune level:" << fc_autotune_level;
  VLOG(5) << "XPU fc autotune file:" << fc_autotune_file;
  VLOG(5) << "XPU fc autotune file writeback:" << fc_autotune_file_writeback;

  if (!fc_autotune_file.empty()) {
    int ret;
    ret = x_context()->set_option("XPU_FC_AUTOTUNE_FILE",
                                  fc_autotune_file.c_str());
    PADDLE_ENFORCE_EQ(
        ret,
        0,
        platform::errors::Unavailable("Failed to set XPU fc autotune file %s.",
                                      fc_autotune_file));
  }
  if (fc_autotune_level > 0) {
    int ret;
    ret = x_context()->set_option("XPU_FC_AUTOTUNE",
                                  (std::to_string(fc_autotune_level)).c_str());
    PADDLE_ENFORCE_EQ(
        ret,
        0,
        platform::errors::Unavailable("Failed to set XPU fc autotune  %d.",
                                      fc_autotune_level));
  }
  if (fc_autotune_file_writeback) {
    int ret;
    ret = x_context()->set_option(
        "XPU_FC_AUTOTUNE_WRITEBACK",
        (std::to_string(fc_autotune_file_writeback)).c_str());
    PADDLE_ENFORCE_EQ(ret,
                      0,
                      platform::errors::Unavailable(
                          "Failed to set XPU fc autotune writeback %d.",
                          fc_autotune_file_writeback));
  }
}

void InferXPUContext::L3CacheAutotune() {
  if (l3_autotune_size_ == 0) return;
  if (holder_map_.empty()) {
    l3_plan_.RunAutotune(l3_blocks_, l3_size_);
    auto* plan = l3_plan_.plan();
    int8_t* cur_l3_ptr = reinterpret_cast<int8_t*>(l3_ptr_);
    for (size_t i = 0; i < l3_blocks_.size(); i++) {
      size_t block_size = plan->at(i);
      VLOG(1) << "ch -- l3_blocks_[" << i << "]:" << l3_blocks_[i]
              << " block_size: " << block_size;
      if (block_size > 0) {
        l3_blocks_[i]->Set(cur_l3_ptr, block_size);
        cur_l3_ptr += block_size;
      }
    }
    VLOG(3) << "========ch -- reset x_context oral================";
    x_context()->_l3_mgr.set(
        reinterpret_cast<int8_t*>(l3_ptr_) + l3_size_ - plan->back(),
        plan->back());
    VLOG(3) << "========ch -- reset x_context oral done================";

    // VLOG(3) << "========ch -- reset x_context================";
    // x_context()->_l3_mgr.set(cur_l3_ptr, plan->back());
    // VLOG(3) << "========ch -- reset x_context done================";

    VLOG(3) << "========ch -- DebugPrint L3 Cache================";
    VLOG(3) << " l3_ptr_ = " << l3_ptr_;
    // VLOG(3) << " reinterpret_cast<int8_t*>(l3_ptr_) = " <<
    // reinterpret_cast<int8_t*>(l3_ptr_);
    VLOG(3) << " l3_size_ = " << l3_size_;
    VLOG(3) << " plan->back() = " << plan->back();
    VLOG(3) << "===========================================";
    VLOG(1) << "ch -- set holder_map by holder_l3_blocks_.";
    DebugPrint();
    for (auto holder_l3_block : holder_l3_blocks_) {
      auto* l3_block = holder_l3_block.second;
      VLOG(1) << "ch -- l3_block : " << l3_block << " l3_block->size()"
              << l3_block->size();
      if (l3_block->size() > 0) {
        auto* holder = holder_l3_block.first;
        auto place = holder->place();
        phi::Allocation* l3_holder =
            new phi::Allocation(l3_block->data(), l3_block->size(), place);
        VLOG(1) << "ch -- add new pair, holders.ptr() : " << holder->ptr()
                << " l3_holder.ptr()" << l3_holder->ptr();
        holder_map_[holder] = std::make_pair(l3_holder, true);
      }
    }
    VLOG(1) << "ch -- after set holder_map by holder_l3_blocks_.";
    DebugPrint();
  } else {
    VLOG(1) << "ch -- L3 cache autotune is already set.";
    DebugPrint();
    for (auto& holders : holder_map_) {
      // VLOG(1) << "ch -- holders.first = " << holders.first->ptr() << ", size"
      // << holders.first->size();
      // VLOG(1) << "ch -- holders.second.first = " <<
      // holders.second.first->ptr()
      // << ", size" << holders.second.first->size();
      // VLOG(1) << "ch -- holders.second.second = " << holders.second.second;
      auto* holder = holders.first;
      auto& holder_pair = holders.second;
      if (!holder_pair.second) {
        swap(*holder, *(holder_pair.first));
        holder_pair.second = true;
      }
    }
    VLOG(1) << "ch -- L3 cache autotune is already set, after swap.";
    DebugPrint();
  }
}
#endif

}  // namespace paddle
