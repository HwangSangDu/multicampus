-- select * from sample.sample21;
use sample;
-- select * from sample21;
-- select no, name from sample21;
-- select *, name from sample31;
-- use sample;
-- select * from sample21;
-- DESC sample21;



-- insert into sample411(no) values(1);
-- select * from sample411;

-- select * from sample41;
-- delete from sample41 where no=2;
-- select distinct name from sample51;
-- select count(distinct name) from sample51;
-- select count(*) from sample51;

-- select * from sample51;
-- select sum(quantity), count(name) from sample51;
-- select count(no), count(distinct name)  from sample51;


-- SELECT 
--     *
-- FROM
--     sample51;
--     
--     
-- SELECT 
--     no, AVG(quantity)
-- FROM
--     sample51
-- group by name;
--     
-- SELECT 
--     AVG(CASE
--         WHEN quantity IS NULL THEN 0
--         else quantity END) AS avg0
-- FROM
--     sample51
-- group by name;
-- 
-- 
-- 


-- use sample; 



-- SELECT 
--     *
-- FROM
--     sample51;
-- 
-- 
-- 
-- -- null은 포함하지 않는다.
-- SELECT 
--     COUNT(*),
--     COUNT(no),
--     COUNT(name),
--     AVG(CASE
--         WHEN quantity IS NULL THEN 0
--         ELSE quantity
--     END) AS average
-- FROM
--     sample51
-- GROUP BY name
-- HAVING average > 1.0
-- order by average desc ;


-- use sakila;

-- select * from sakila.film_category;

-- select category_id, count(category_id) as c_count
-- from sakila.film_category group by category_id order by c_count desc;

 
-- select name, count(name)
-- from sample.sample51
-- -- where count(name)=1
-- group by name
-- having count(name) = 1;

--  집계함수 이후 조건문은 having 절이 필요하다.




use sample;
select * from sample54;





/*
crtl + shift + enter  (다중 쿼리 실행)
crtl + /  (주석)
crtl + B  (정렬)
crtl + shift + o (.sql 열기)
*/ 

