/*
  Copyright (C) 2018  Wennie Tabib

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef GMM_2_BASE_H
#define GMM_2_BASE_H

#include <Eigen/Core>
#include <Eigen/Dense>

#include <gmm/GMMNBase.h>

namespace gmm_utils
{
  template <typename T, int K=-1, int S=-1>
    class GMM2Base : public GMMNBase<T,2,K,S>
    {
      public:

      using Ptr = std::shared_ptr<GMM2Base<T,K,S> >;
      using ConstPtr = std::shared_ptr<GMM2Base<T,K,S> const>;

      GMM2Base() : GMMNBase<T,2,K,S>() {}
      GMM2Base(const GMM2Base &d) : GMMNBase<T,2,K,S>(d) {}
      GMM2Base(const GMMNBase<T,2,K,S> &d) : GMMNBase<T,2,K,S>(d) {}
    };

  typedef GMM2Base<float,-1,-1> GMM2f;
  typedef GMM2Base<double,-1,-1> GMM2d;
}



#endif // GMM_2_BASE_H
