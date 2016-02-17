#ifndef CNN_TRAINING_H_
#define CNN_TRAINING_H_

#include <vector>
#include "cnn/model.h"
#include "cnn/shadow-params.h"

namespace cnn {

struct Trainer {
  explicit Trainer(Model* m, cnn::real lam, cnn::real e0) :
    eta0(e0), eta(e0), eta_decay(), epoch(), lambda(lam), clipping_enabled(true), clip_threshold(5), clips(), updates(), model(m) {}
  virtual ~Trainer();

  virtual void update(cnn::real nutt = 1.0, cnn::real scale = 1.0) = 0;
  void update_epoch(cnn::real r = 1) {
    epoch += r;
    eta = eta0 / (1 + epoch * eta_decay);
  }

  // if clipping is enabled and the gradient is too big, return the amount to
  // scale the gradient by (otherwise 1)
  /**
  @nutt: proportional to the number of utterances trained in parallel
  */
  cnn::real clip_gradients(cnn::real nutt = 1.0);

  // learning rates
  cnn::real eta0;
  cnn::real eta;
  cnn::real eta_decay;
  cnn::real epoch;

  cnn::real lambda; // weight regularization (l2)

  // clipping
  cnn::real clipping_enabled;
  cnn::real clip_threshold;
  cnn::real clips;
  cnn::real updates;

  void status() {
    std::cerr << "[epoch=" << epoch << " eta=" << eta << " clips=" << clips << " updates=" << updates << "] ";
    updates = clips = 0;
  }

  Model* model;  // parameters and gradients live here
};

struct SimpleSGDTrainer : public Trainer {
    explicit SimpleSGDTrainer(Model* m, cnn::real lam = 1e-6, cnn::real e0 = 0.1) : Trainer(m, lam, e0) {}
    void update(cnn::real nutt, cnn::real scale = 1.0) override;
    void update(const std::vector<LookupParameters*> &lookup_params, const std::vector<Parameters*> &params, cnn::real nutt = 1.0, cnn::real scale = 1);
};

/** normalized gradient descent trainer
according to the paper 
Beyond Convexity: Stochastic Quasi-Convex Optimization @ NIPS 2015
Elad Hazan, Princeton University; Kfir Levy*, Technion; Shai Shalev-Shwartz, Hebrew University
todo
struct NGDTrainer : public Trainer {
    explicit NGDTrainer(Model* m, cnn::real lam = 1e-6, cnn::real e0 = 0.1) : Trainer(m, lam, e0) {}
    void update(cnn::real nutt, cnn::real scale) override;
    void update(const std::vector<LookupParameters*> &lookup_params, const std::vector<Parameters*> &params, cnn::real nutt = 1.0, cnn::real scale = 1);
};
*/

struct MomentumSGDTrainer : public Trainer {
  explicit MomentumSGDTrainer(Model* m, cnn::real lam = 1e-6, cnn::real e0 = 0.01, cnn::real mom = 0.9) :
    Trainer(m, lam, e0), momentum(mom), velocity_allocated(false) {}
  void update(cnn::real nutt, cnn::real scale = 1.0) override;

  cnn::real momentum;

  bool velocity_allocated;

  // the following represent the current velocity
  std::vector<ShadowParameters> vp;
  std::vector<ShadowLookupParameters> vlp;
  //std::unordered_map<Parameters*, Tensor> vp;
  //std::unordered_map<LookupParameters*, std::unordered_map<unsigned, Tensor>> vl;
};

struct AdagradTrainer : public Trainer {
  explicit AdagradTrainer(Model* m, cnn::real lam = 1e-6, cnn::real e0 = 0.1, cnn::real eps = 1e-20) :
    Trainer(m, lam, e0), epsilon(eps), shadow_params_allocated(false) {}
  void update(cnn::real nutt, cnn::real scale = 1.0) override;

  cnn::real epsilon;
  bool shadow_params_allocated;
  std::vector<ShadowParameters> vp;
  std::vector<ShadowLookupParameters> vlp;
};

struct AdadeltaTrainer : public Trainer {
  explicit AdadeltaTrainer(Model* m, cnn::real lam = 1e-6, cnn::real eps = 1e-6, cnn::real rho = 0.95) :
    Trainer(m, lam, 1.0), epsilon(eps), rho(rho), shadow_params_allocated(false) {}
  void update(cnn::real nutt, cnn::real scale = 1.0) override;

  cnn::real epsilon;
  cnn::real rho;
  bool shadow_params_allocated;
  std::vector<ShadowParameters> hg; // History of gradients
  std::vector<ShadowLookupParameters> hlg;
  std::vector<ShadowParameters> hd; // History of deltas
  std::vector<ShadowLookupParameters> hld;
};

struct RmsPropTrainer : public Trainer {
  explicit RmsPropTrainer(Model* m, cnn::real lam = 1e-6, cnn::real e0 = 0.1, cnn::real eps = 1e-20, cnn::real rho = 0.95) :
    Trainer(m, lam, e0), epsilon(eps), rho(rho), shadow_params_allocated(false) {}
  void update(cnn::real nutt, cnn::real scale = 1.0) override;

  cnn::real epsilon;
  cnn::real rho;
  bool shadow_params_allocated;
  std::vector<cnn::real> hg; // History of gradients
  std::vector<std::vector<cnn::real> > hlg;
};

/**
In some cases, adding a momentum term β is beneficial. Here, Nesterov momentum is used:
See descriptions in http://climin.readthedocs.org/en/latest/rmsprop.html
*/
struct RmsPropWithMomentumTrainer : public Trainer {
    explicit RmsPropWithMomentumTrainer(Model* m, cnn::real lam = 1e-6, cnn::real e0 = 0.1, cnn::real eps = 1e-20, cnn::real rho = 0.95, cnn::real mom = 0.9) :
        Trainer(m, lam, e0), epsilon(eps), rho(rho), shadow_params_allocated(false), momentum(mom) {}
    void update(cnn::real nutt, cnn::real scale = 1.0) override;

    cnn::real epsilon;
    cnn::real rho;
    bool shadow_params_allocated;
    std::vector<cnn::real> hg; // History of gradients
    std::vector<std::vector<cnn::real> > hlg;

    cnn::real momentum;
    // the following represent the current velocity
    std::vector<ShadowParameters> vp;
    std::vector<ShadowLookupParameters> vlp;
};

struct AdamTrainer : public Trainer {
  explicit AdamTrainer(Model* m, cnn::real lambda = 1e-6, cnn::real alpha = 0.001, cnn::real beta_1 = 0.9, cnn::real beta_2 = 0.999, cnn::real eps = 1e-8) :
    Trainer(m, lambda, alpha), beta_1(beta_1), beta_2(beta_2), eps(eps), shadow_params_allocated(false) {}

  void update(cnn::real nutt, cnn::real scale = 1.0) override;

  cnn::real beta_1;
  cnn::real beta_2;
  cnn::real eps;
  bool shadow_params_allocated;
  std::vector<ShadowParameters> m; // History of gradients
  std::vector<ShadowLookupParameters> lm;
  std::vector<ShadowParameters> v; // History of deltas
  std::vector<ShadowLookupParameters> lv;
};

} // namespace cnn

#endif
