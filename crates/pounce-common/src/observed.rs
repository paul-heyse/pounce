//! Synchronous worker-local operation admission and observation.
//! Numerical owners remain responsible for arithmetic. An observer failure is sticky and
//! terminal until the scope drops; it must never be converted into numerical fallback.
use std::{cell::RefCell, panic::{catch_unwind, AssertUnwindSafe}, rc::Rc};
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Abort { Cancelled, Resource(String), Contract(String), Panic }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Path { Monolithic, SchurF, SchurS }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Primitive { Factor, Backsolve }
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub x: Vec<usize>, pub c: Vec<usize>, pub d: Vec<usize>,
    pub full_to_x: Vec<i32>, pub full_to_c: Vec<i32>, pub full_to_d: Vec<i32>,
    pub fixed_removed: Vec<usize>, pub relaxed_fixed: bool,
}
impl Layout {
    pub fn dimension(&self) -> usize { self.x.len() + self.c.len() + 2 * self.d.len() }
    pub fn x_index(&self, original: usize) -> Option<usize> { self.full_to_x.get(original).copied().filter(|i| *i >= 0).map(|i| i as usize) }
    pub fn row_index(&self, original: usize) -> Option<usize> {
        let c = *self.full_to_c.get(original)?;
        if c >= 0 { Some(self.x.len() + self.d.len() + c as usize) }
        else { let d = *self.full_to_d.get(original)?; if d>=0 {Some(self.x.len() + self.d.len() + self.c.len() + d as usize)} else {None} }
    }
}
#[derive(Clone, Debug)]
pub enum StorageScope { Application, Linear }
#[derive(Clone, Debug)]
pub enum Event {
    /// Before allocation. Extent describes known owner buffers; opaque=true is an
    /// explicit request for a complete foreign reservation, never a complete estimate.
    Storage { scope: StorageScope, owner: &'static str, instance: usize, known_bytes: usize, opaque: bool },
    Begin { path: Path, primitive: Primitive, rhs: usize, refinement_bound: usize },
    End { path: Path, primitive: Primitive, succeeded: bool },
    Selected { schur: bool, reason: &'static str },
}
pub trait Observer {
    fn bind_layout(&self, layout: &Layout) -> Result<Option<Vec<usize>>, Abort>;
    fn event(&self, event: &Event) -> Result<(), Abort>;
}
#[derive(Default)]
struct State { observer: Option<Rc<dyn Observer>>, abort: Option<Abort>, linear_maximum: Option<usize>, linear_bounded: bool }
thread_local! { static ACTIVE: RefCell<State> = RefCell::new(State::default()); }
pub struct Scope { previous: Option<State> }
impl Scope {
    pub fn enter(observer: Rc<dyn Observer>) -> Self {
        let previous = ACTIVE.with(|state| state.replace(State { observer: Some(observer), abort: None, linear_maximum: None, linear_bounded: false }));
        Self { previous: Some(previous) }
    }
    pub fn abort(&self) -> Option<Abort> { abort() }
}
impl Drop for Scope { fn drop(&mut self) { if let Some(previous) = self.previous.take() { ACTIVE.with(|state| {state.replace(previous);}); } } }
pub fn set_linear_bounded(bounded:bool) { ACTIVE.with(|state|state.borrow_mut().linear_bounded=bounded); }
pub fn linear_bounded()->bool {ACTIVE.with(|state|state.borrow().linear_bounded)}
pub fn set_linear_maximum(maximum:Option<usize>) { ACTIVE.with(|state|state.borrow_mut().linear_maximum=maximum); }
pub fn linear_maximum()->Option<usize> {ACTIVE.with(|state|state.borrow().linear_maximum)}
pub fn reject(error:Abort) { ACTIVE.with(|state| {let mut state=state.borrow_mut();if state.abort.is_none(){state.abort=Some(error);}}); }
pub fn abort() -> Option<Abort> { ACTIVE.with(|state| state.borrow().abort.clone()) }
fn invoke<T>(allow_after_abort: bool, f: impl FnOnce(&dyn Observer) -> Result<T, Abort>, default: T) -> Result<T, Abort> {
    let (observer, stopped) = ACTIVE.with(|state| { let state=state.borrow(); (state.observer.clone(),state.abort.clone()) });
    if !allow_after_abort { if let Some(stopped)=stopped { return Err(stopped); } }
    let Some(observer)=observer else { return Ok(default); };
    let result=catch_unwind(AssertUnwindSafe(|| f(observer.as_ref()))).unwrap_or(Err(Abort::Panic));
    if let Err(error)=&result { reject(error.clone()); }
    result
}
pub fn event(event: Event) -> Result<(), Abort> { invoke(matches!(event,Event::End {..}), |observer| observer.event(&event), ()) }
pub fn bind_layout(layout: &Layout) -> Result<Option<Vec<usize>>, Abort> { invoke(false, |observer| observer.bind_layout(layout), None) }
/// Owner-side scope: admission precedes arithmetic; failed early returns remain observed.
pub struct Operation { path: Path, primitive: Primitive, ended: bool }
impl Operation {
    pub fn begin(path: Path, primitive: Primitive, rhs: usize, refinement_bound: usize) -> Result<Self, Abort> {
        event(Event::Begin { path, primitive, rhs, refinement_bound })?;
        Ok(Self { path, primitive, ended: false })
    }
    pub fn finish(mut self, succeeded: bool) -> bool {
        self.ended=true;
        event(Event::End { path:self.path, primitive:self.primitive, succeeded }).is_ok()
    }
}
impl Drop for Operation { fn drop(&mut self) { if !self.ended { let _=event(Event::End { path:self.path, primitive:self.primitive, succeeded:false }); } } }

#[cfg(test)]
mod tests {
    use super::*;
    struct Deny;
    impl Observer for Deny {
        fn bind_layout(&self, _: &Layout) -> Result<Option<Vec<usize>>, Abort> { Ok(None) }
        fn event(&self, _: &Event) -> Result<(), Abort> { Err(Abort::Resource("pool".into())) }
    }
    #[test]
    fn wrapper_storage_overflow_and_refusal_are_terminal_before_work() {
        let scope=Scope::enter(Rc::new(Deny));
        set_linear_bounded(true);
        assert!(!reserve_wrapper("woodbury",1,usize::MAX,1,0));
        assert!(matches!(scope.abort(),Some(Abort::Resource(_))));
        assert!(Operation::begin(Path::Monolithic,Primitive::Factor,0,0).is_err());
        drop(scope);
        let scope=Scope::enter(Rc::new(Deny));
        set_linear_bounded(true);
        assert!(!reserve_wrapper("restoration",2,12,4,24));
        assert!(scope.abort().is_some());
    }
    #[test]
    fn admission_abort_is_sticky_and_next_attempt_resets_it() {
        let scope=Scope::enter(Rc::new(Deny));
        assert!(Operation::begin(Path::SchurF,Primitive::Factor,0,0).is_err());
        assert!(scope.abort().is_some());
        assert!(event(Event::Selected {schur:false,reason:"fallback"}).is_err());
        drop(scope);
        assert!(abort().is_none());
    }
}

/// Reserve simultaneously live dense/multivector wrapper owners before construction.
/// Includes old/new copies, rank products, projected triplets, action scratch and headers.
pub fn reserve_wrapper(owner:&'static str,instance:usize,n:usize,rank:usize,triplets:usize)->bool {
    let extent=(|| {let scalars=n.checked_mul(n)?.checked_mul(128)?
        .checked_add(n.checked_mul(rank)?.checked_mul(128)?)?
        .checked_add(rank.checked_mul(rank)?.checked_mul(64)?)?
        .checked_add(triplets.checked_mul(128)?)?.checked_add(n.checked_mul(256)?)?.checked_add(1024)?;
        scalars.checked_mul(2*std::mem::size_of::<usize>())}) ();
    let bounded=linear_bounded();
    if bounded && extent.is_none(){reject(Abort::Resource("linear wrapper storage extent overflow".into()));return false;}
    event(Event::Storage {scope:StorageScope::Linear,owner,instance,known_bytes:extent.unwrap_or(0),opaque:!bounded}).is_ok()
}
