#include "cnn/training.h"

#include "cnn/gpu-ops.h"

namespace cnn {

using namespace std;

template <class Derived>
bool is_valid(const Eigen::MatrixBase<Derived>& x) {
  return ((x - x).array() == (x - x).array()).all();
}

Trainer::~Trainer() {}

/** 
@scale : proportional to the number of samples trained in parallel 
*/
cnn::real Trainer::clip_gradients(cnn::real samples) {
  cnn::real gscale = 1;
  if (clipping_enabled) {
    cnn::real gg = model->gradient_l2_norm();
    if (gg > clip_threshold * samples) {
      ++clips;
      gscale = (clip_threshold * samples) / gg;
    }
  }
  return gscale;
}

void SimpleSGDTrainer::update(cnn::real nutt, cnn::real scale) {
    update(model->lookup_parameters_list(), model->parameters_list(), nutt, scale);
}

void SimpleSGDTrainer::update(const std::vector<LookupParameters*> &lookup_params, const std::vector<Parameters*> &params, cnn::real samples, cnn::real scale) {
  const cnn::real gscale = clip_gradients(samples);
  cnn::real nutt_scale = 1.0 / samples;
  for (auto p : params) {
#if HAVE_CUDA
    gpu::sgd_update(p->values.d.size(), p->g.v, p->values.v, eta * scale * gscale * nutt_scale, lambda);
#else
    auto reg = (*p->values) * lambda;
    *p->values -= nutt_scale * (eta * scale * gscale) * *p->g + reg;
#endif
    p->clear();
  }
  for (auto p : lookup_params) {
    for (auto i : p->non_zero_grads) {
#if HAVE_CUDA
      gpu::sgd_update(p->values[i].d.size(), p->grads[i].v, p->values[i].v, eta * scale * gscale * nutt_scale, lambda);
#else
      auto reg = (*p->values[i]) * lambda;
      *p->values[i] -= *p->grads[i] * (eta * scale * gscale * nutt_scale) + reg;
#endif
    }
    p->clear();
  }
  ++updates;
}

void MomentumSGDTrainer::update(cnn::real nutt, cnn::real scale) {
  // executed on the first iteration to create vectors to
  // store the velocity
  if (!velocity_allocated) {
    vp = AllocateShadowParameters(*model);
    vlp = AllocateShadowLookupParameters(*model);
    velocity_allocated = true;
  }

  const cnn::real gscale = clip_gradients(nutt);
  cnn::real nutt_scale = 1.0 / nutt;
  unsigned pi = 0;
  for (auto p : model->parameters_list()) {
    Tensor& v = vp[pi++].h;
#if HAVE_CUDA
    gpu::sgd_momentum_update(p->values.d.size(), p->g.v, p->values.v, v.v, eta * scale * gscale * nutt_scale, lambda, momentum);
#else
    auto reg = *p->values * lambda;
    (*v) = momentum * (*v) - (eta * scale * gscale*nutt_scale) * (*p->g);
//    if (verbose)
//    {
//        cout << "name= " << p->name << " v= " << p->values.v[0] << " g= " << p->g.v[0] << " dv=" << v.v[0] << endl;
//    }
    *p->values += *v - reg;
#endif
    p->clear();
  }
  pi = 0;
  for (auto p : model->lookup_parameters_list()) {
    vector<Tensor>& vx = vlp[pi++].h;
    for (auto i : p->non_zero_grads) {
#if HAVE_CUDA
        gpu::sgd_momentum_update(p->values[i].d.size(), p->grads[i].v, p->values[i].v, vx[i].v, eta * scale * gscale * nutt_scale, lambda, momentum);
#else
      Tensor& v = vx[i];
      auto reg = (*p->values[i]) * lambda;
      (*v) = momentum * (*v) - (eta * scale * gscale*nutt_scale) * (*p->grads[i]);
      *p->values[i] += *v - reg;
#endif
    }
    p->clear();
  }
  ++updates;
}

void AdagradTrainer::update(cnn::real nsamples, cnn::real scale) {
  unsigned pi;
  if (!shadow_params_allocated) {
    vp = AllocateShadowParameters(*model);
    vlp = AllocateShadowLookupParameters(*model);
    shadow_params_allocated = true;
  }

  pi = 0;
  const cnn::real gscale = clip_gradients(nsamples);
  for (auto p : model->parameters_list()) {
    Tensor& v = vp[pi++].h;
    auto reg = (*p->values) * lambda;
    auto g2 = (*p->g).cwiseProduct(*p->g);
    (*v) += g2;
    auto delta = -(eta * scale * gscale) * (*p->g).cwiseQuotient(((*v).array() + epsilon).matrix().cwiseSqrt());
    *p->values += delta - reg;
    p->clear();
  }

  pi = 0;
  for (auto p : model->lookup_parameters_list()) {
    vector<Tensor>& vx = vlp[pi++].h;
    for (auto i : p->non_zero_grads) {
      Tensor& v = vx[i];
      auto reg = (*p->values[i]) * lambda;
      auto g2 = (*p->grads[i]).cwiseProduct(*p->grads[i]);
      (*v) += g2;
      auto delta = -(eta * scale * gscale) * (*p->grads[i]).cwiseQuotient(((*v).array() + epsilon).matrix().cwiseSqrt());
      *p->values[i] += delta - reg;
    }
    p->clear();
  }

  ++updates;
}

void AdadeltaTrainer::update(cnn::real nutt, cnn::real scale) {
  unsigned pi;
  if (!shadow_params_allocated) {
    hg = AllocateShadowParameters(*model);
    hlg = AllocateShadowLookupParameters(*model);
    hd = AllocateShadowParameters(*model);
    hld = AllocateShadowLookupParameters(*model);

    /*pi = 0;
    for (auto p : model->parameters_list()) {
      TensorTools::Constant(hg[pi].h, epsilon);
      TensorTools::Constant(hd[pi].h, epsilon);
      ++pi;
    }

    pi = 0;
    for (auto p : model->lookup_parameters_list()) {
      vector<Tensor>& hgx = hlg[pi].h;
      vector<Tensor>& hdx = hld[pi].h;
      for (unsigned i = 0; i < hgx.size(); ++i) {
        TensorTools::Constant(hgx[i], epsilon);
        TensorTools::Constant(hdx[i], epsilon);
      }
      ++pi;
    }*/

    shadow_params_allocated = true;
  }

  const cnn::real gscale = clip_gradients(nutt);
  cnn::real nutt_scale = 1.0 / nutt;
  pi = 0;
  for (auto p : model->parameters_list()) {
    auto& g = (scale * gscale * nutt_scale) * *p->g;
    Tensor& hgv = hg[pi].h;
    Tensor& hdv = hd[pi].h;
    auto reg = (*p->values) * lambda;
    auto g2 = g.cwiseProduct(g);
    *hgv = rho * *hgv + (1.0 - rho) * g2;
    auto num = -g.cwiseProduct(((*hdv).array() + epsilon).matrix().cwiseSqrt());
    auto den = ((*hgv).array() + epsilon).matrix().cwiseSqrt();
    auto delta = num.cwiseQuotient(den);
    auto d2 = delta.cwiseProduct(delta);
    *hdv = rho * *hdv + (1.0 - rho) * d2;
    *p->values += delta - reg;
    p->clear();
    pi++;
  }

  pi = 0;
  for (auto p : model->lookup_parameters_list()) {
    vector<Tensor>& hgvx = hlg[pi].h;
    vector<Tensor>& hdvx = hld[pi].h;
    for (auto i : p->non_zero_grads) {
      Tensor& hgv = hgvx[i];
      Tensor& hdv = hdvx[i];
      auto& g = scale * gscale * nutt_scale * *p->grads[i];
      auto reg = (*p->values[i]) * lambda;
      auto g2 = g.cwiseProduct(g);
      *hgv = rho * *hgv + (1.0 - rho) * g2;
      auto num = -g.cwiseProduct(((*hdv).array() + epsilon).matrix().cwiseSqrt());
      auto den = ((*hgv).array() + epsilon).matrix().cwiseSqrt();
      auto delta = num.cwiseQuotient(den);
      auto d2 = delta.cwiseProduct(delta);
      *hdv = rho * *hdv + (1.0 - rho) * d2;
      *p->values[i] += delta - reg;
    }
    p->clear();
    pi++;
  }
  ++updates;
}

void RmsPropTrainer::update(cnn::real nutt, cnn::real scale) {
  unsigned pi = 0;
  if (!shadow_params_allocated) {
    hg.resize(model->parameters_list().size());

    pi = 0;
    hlg.resize(model->lookup_parameters_list().size());
    for (auto p : model->lookup_parameters_list()) {
      hlg[pi++].resize(p->size());
    }

    shadow_params_allocated = true;
  }

  const cnn::real gscale = clip_gradients(nutt);
  cnn::real nutt_scale = 1.0 / nutt;
  pi = 0;
  for (auto p : model->parameters_list()) {
    cnn::real& d2 = hg[pi++];
    auto reg = (*p->values) * lambda;
    cnn::real g2 = (*p->g).squaredNorm();
    d2 = rho * d2 + (1.0 - rho) * g2;
    *p->values -= ((eta * scale * gscale * nutt_scale / sqrt(d2 + epsilon)) * *p->g + reg);
    p->clear();
  }

  pi = 0;
  for (auto p : model->lookup_parameters_list()) {
    vector<cnn::real>& hlgx = hlg[pi++];
    for (auto i : p->non_zero_grads) {
      cnn::real& d2 = hlgx[i];
      auto reg = (*p->values[i]) * lambda;
      cnn::real g2 = (*p->grads[i]).squaredNorm();
      d2 = rho * d2 + (1.0 - rho) * g2;
      *p->values[i] -= ((eta * scale * gscale * nutt_scale / sqrt(d2 + epsilon)) * *p->grads[i] + reg);
    }
    p->clear();
  }
  ++updates;
}

void RmsPropWithMomentumTrainer::compute_gradient_norm(
    std::vector<Parameters*> plist, std::vector<cnn::real>& vpgrd_norm,
    std::vector<LookupParameters*> llist, std::vector<cnn::real>& vl_grd_norm) 
{
    vector<vector<cnn::real>> v_norm(2);

    /// get the number of parameters for parm and lookup_param
    vector<int> i_mdl_size(2, plist.size());
    i_mdl_size[1] = 0;
    for (auto p : llist) {
        for (auto i : p->non_zero_grads) i_mdl_size[1]++;
    }
    
    for (int k = 0; k < 2; k++)
    {
        v_norm[k].resize(i_mdl_size[k]);
    }
#if HAVE_CUDA
    cnn::real * ptr_gnorm_param[2];
    for (int k = 0; k < 2; k++)
    {
        CUDA_CHECK(cudaMalloc(&ptr_gnorm_param[k], sizeof(cnn::real)*i_mdl_size[k]));
    }
#endif
    int pi = 0;
    for (auto p : plist) {
#if HAVE_CUDA
        gpu::l2_norm_reducer(p->g.d.size(), p->g.v, &ptr_gnorm_param[0][pi], true, false);
#else
        cnn::real g2 = (*p->g).squaredNorm();
        v_norm[0][pi] = g2;
#endif
        pi++;
    }

    pi = 0;
    for (auto p : llist) {
        for (auto i : p->non_zero_grads) {
#if HAVE_CUDA
            gpu::l2_norm_reducer(p->grads[i].d.size(), p->grads[i].v, &ptr_gnorm_param[1][pi], true, false);
#else
            cnn::real g2 = (*p->grads[i]).squaredNorm();
            v_norm[1][pi] = g2;
#endif
            pi++;
        }
    }

#if HAVE_CUDA
    for (int k = 0; k < 2; k++)
    {
        CUDA_CHECK(cudaMemcpyAsync(&v_norm[k][0], ptr_gnorm_param[k], sizeof(cnn::real)*i_mdl_size[k], cudaMemcpyDeviceToHost));
    }
    for (int k = 0; k < 2; k++)
    {
        CUDA_CHECK(cudaFree(ptr_gnorm_param[k]));
    }
#endif

    vpgrd_norm = v_norm[0];
    vl_grd_norm = v_norm[1];
}

void RmsPropWithMomentumTrainer::update(cnn::real nutt, cnn::real scale) {
    unsigned pi = 0;
    if (!shadow_params_allocated) {
        hg.resize(model->parameters_list().size());

        pi = 0;
        hlg.resize(model->lookup_parameters_list().size());
        for (auto p : model->lookup_parameters_list()) {
            hlg[pi++].resize(p->size());
        }

        vp = AllocateShadowParameters(*model);
        vlp = AllocateShadowLookupParameters(*model);

        shadow_params_allocated = true;
    }

    /// compute norm of gradients
    vector<cnn::real> vpgrd_norm;
    vector<cnn::real> vlgrd_norm;
    compute_gradient_norm(model->parameters_list(), vpgrd_norm, model->lookup_parameters_list(), vlgrd_norm);


    const cnn::real gscale = clip_gradients(nutt);
    cnn::real nutt_scale = 1.0 / nutt;
    pi = 0;
    for (auto p : model->parameters_list()) {
        cnn::real& d2 = hg[pi];
        Tensor& v = vp[pi].h;
#if HAVE_CUDA
        gpu::rmsprop_momentum_update(p->values.d.size(), p->g.v, p->values.v, v.v, &d2, eta * scale * gscale, lambda, momentum, rho, epsilon, vpgrd_norm[pi]);
#else
        auto reg = (*p->values) * lambda;
        cnn::real g2 = vpgrd_norm[pi];
        d2 = rho * d2 + (1.0 - rho) * g2;

        (*v) = momentum * (*v) - (eta * scale * gscale / sqrt(d2 + epsilon)) * *p->g;

        *p->values += *v - reg; 
#endif
        pi++;
        p->clear();
    }

    pi = 0;
    int li = 0; 
    for (auto p : model->lookup_parameters_list()) {
        vector<cnn::real>& hlgx = hlg[pi];
        vector<Tensor>& vx = vlp[pi++].h;
        for (auto i : p->non_zero_grads) {
            Tensor& v = vx[i];
            cnn::real& d2 = hlgx[i];
#if HAVE_CUDA
            gpu::rmsprop_momentum_update(p->values[i].d.size(), p->grads[i].v, p->values[i].v, v.v, &d2, eta * scale * gscale, lambda, momentum, rho, epsilon, vlgrd_norm[li]);
#else
            auto reg = (*p->values[i]) * lambda;
            cnn::real g2 = vlgrd_norm[li];
            d2 = rho * d2 + (1.0 - rho) * g2;
            (*v) = momentum * (*v) - (eta * scale * gscale  / sqrt(d2 + epsilon)) * *p->grads[i];
            *p->values[i] += *v - reg; 
#endif
            li++;
        }
        p->clear();
    }

    ++updates;
}

void AdamTrainer::update(cnn::real nutt, cnn::real scale) {
  unsigned pi;
  if (!shadow_params_allocated) {
    m = AllocateShadowParameters(*model);
    lm = AllocateShadowLookupParameters(*model);
    v = AllocateShadowParameters(*model);
    lv = AllocateShadowLookupParameters(*model);
    shadow_params_allocated = true;
  }

  const cnn::real gscale = clip_gradients();
  cnn::real nutt_scale = 1.0 / nutt;
  pi = 0;
  static unsigned t = 0;
  for (auto p : model->parameters_list()) {
    ++t;
    auto g_t = (scale * gscale * nutt_scale) * *p->g;
    auto m_t = *m[pi].h;
    auto v_t = *v[pi].h;
    auto reg = (*p->values) * lambda;
    m_t = beta_1 * m_t + (1 - beta_1) * g_t;
    auto g2 = g_t.cwiseProduct(g_t);
    v_t = beta_2 * v_t + (1 - beta_2) * g2;
    cnn::real s1 = 1 - pow(beta_1, t);
    cnn::real s2 = 1 - pow(beta_2, t);
    auto mhat = m_t / s1;
    auto vhat = v_t / s2;
    auto delta = (-eta * mhat).cwiseQuotient((vhat.array().sqrt() + eps).matrix());
    *p->values += delta - reg;
    p->clear();
    pi++;
  }

  pi = 0;
  for (auto p : model->lookup_parameters_list()) {
    vector<Tensor>& vm = lm[pi].h;
    vector<Tensor>& vv = lv[pi].h;
    for (auto i : p->non_zero_grads) {
      auto m_t = *vm[i];
      auto v_t = *vv[i];
      auto g_t = scale * gscale * nutt_scale * *p->grads[i];
      auto g2 = g_t.cwiseProduct(g_t);
      auto reg = (*p->values[i]) * lambda;
      m_t = beta_1 * m_t + (1 - beta_1) * g_t;
      v_t = beta_2 * v_t + (1 - beta_2) * g2;
      cnn::real s1 = 1 - pow(beta_1, t);
      cnn::real s2 = 1 - pow(beta_2, t);
      auto mhat = m_t / s1;
      auto vhat = v_t / s2;
      auto delta = (-eta * mhat).cwiseQuotient((vhat.array().sqrt() + eps).matrix());
      *p->values[i] += delta - reg;
    }
    p->clear();
    pi++;
  }
  ++updates;
}

} // namespace cnn
