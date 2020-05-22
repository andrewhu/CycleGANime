import Vue from 'vue'
import VueRouter from 'vue-router'
import Home from '../pages/Home.vue'
import Result from '../pages/Result.vue'
import Sample from '../pages/Sample.vue'
import NotFound from '../pages/NotFound.vue'

Vue.use(VueRouter)

const routes = [
    {
        path: '/',
        name: 'Home',
        component: Home
    },
    {
        path: '/samples',
        name: 'samples',
        component: Sample
    },
    {
        path: '/404',
        name: "NotFound",
        component: NotFound
    },
    {
      path: '/:id',
        name: "results",
        component: Result
    },
]

const router = new VueRouter({
    mode: 'history',
    base: process.env.BASE_URL,
    routes
})

export default router
