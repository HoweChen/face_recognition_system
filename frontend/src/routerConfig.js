// 以下文件格式为描述路由的协议格式
// 你可以调整 routerConfig 里的内容
// 变量名 routerConfig 为 iceworks 检测关键字，请不要修改名称

import BlankLayout from './layouts/BlankLayout';
import Home from './pages/Home';
import Register from './pages/Register';
import Login from './pages/Login';

import Console from './pages/Console';
import NotFound from './pages/NotFound';

const routerConfig = [
  {
    path: '/',
    layout: BlankLayout,
    component: Home,
  },
  {
    path: '/register',
    layout: BlankLayout,
    component: Register,
  },
  {
    path: '/login',
    layout: BlankLayout,
    component: Login,
  },
  {
    path: '/console',
    layout: BlankLayout,
    component: Console,
  },
  {
    path: '*',
    layout: BlankLayout,
    component: NotFound,
  },
];

export default routerConfig;
